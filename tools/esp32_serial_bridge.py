#!/usr/bin/env python3
"""
VoxFlow ESP32 serial bridge.

Supports both:
  1. Serial-start mode (default, automated capture trigger)
  2. Physical-buttons mode (--physical-buttons, waits for ESP32 GPIO27 START)

VXF1 Continuous Streaming Flow:
    ESP32 + INMP441 (Producer)
        -> Bounded Ring Buffer
        -> USB Serial @ 460800 baud (Consumer)
        -> Python Bridge (VXF1 Framed Stream Receiver)
        -> Verified Continuous PCM (3.0s - 60.0s via physical STOP / serial STOP / timeout)
        -> WAV file
        -> existing VoxFlow REST API
        -> existing HuBERT/session pipeline

Binary audio is NEVER parsed with readline().
Every binary frame is validated for Magic, Sequence, Length, and CRC32.
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import time
import wave
import zlib
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import requests
import serial
from serial import SerialException


DEFAULT_PORT = os.getenv("VOXFLOW_SERIAL_PORT", "COM8")
DEFAULT_BAUD = int(os.getenv("VOXFLOW_BAUD_RATE", "460800"))
DEFAULT_API = os.getenv("VOXFLOW_API_URL", "http://127.0.0.1:5000")
DEFAULT_USER = os.getenv("VOXFLOW_USER_ID", "default_user")

READLINE_TIMEOUT = 8.0
CAPTURE_HEADER_TIMEOUT = 75.0
FRAME_TIMEOUT = 10.0
ANALYZE_TIMEOUT = 1800.0

FRAME_HEADER_MAGIC = b"VXF1"
FRAME_HEADER_SIZE = 16  # 4 bytes magic + 4 bytes seq + 4 bytes length + 4 bytes crc32
FRAME_HEADER_FORMAT = "<4sIII"


class BridgeError(RuntimeError):
    """Expected hardware / bridge failure."""


def read_status_line(ser: serial.Serial, timeout: float) -> str:
    """Read one ASCII status line with a wall-clock timeout."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        line = ser.readline()
        if not line:
            continue

        text = line.decode("ascii", errors="replace").strip()
        if text:
            print(f"[ESP32] {text}")
        return text

    raise BridgeError("Timed out waiting for an ESP32 status line.")


def wait_for_exact_line(ser: serial.Serial, expected: str, timeout: float) -> None:
    """Ignore unrelated status lines until the expected line arrives."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        ser.timeout = min(0.5, remaining)
        line = ser.readline()
        if not line:
            continue

        text = line.decode("ascii", errors="replace").strip()
        if text:
            print(f"[ESP32] {text}")

        if text == expected:
            return

        if text.startswith("ERROR:"):
            raise BridgeError(f"ESP32 reported {text}")

    raise BridgeError(f"Timed out waiting for '{expected}'.")


def wait_for_prefix(ser: serial.Serial, prefix: str, timeout: float) -> str:
    """Ignore unrelated status lines until a line with the given prefix arrives."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        ser.timeout = min(0.5, remaining)
        line = ser.readline()
        if not line:
            continue

        text = line.decode("ascii", errors="replace").strip()
        if text:
            print(f"[ESP32] {text}")

        if text.startswith(prefix):
            return text

        if text.startswith("ERROR:"):
            raise BridgeError(f"ESP32 reported {text}")

    raise BridgeError(f"Timed out waiting for a line beginning with '{prefix}'.")


def read_exact(stream: Union[serial.Serial, BinaryIO], size: int, timeout: float) -> bytes:
    """Read exactly size binary bytes from serial or stream without line parsing."""
    if size == 0:
        return b""

    data = bytearray()
    deadline = time.monotonic() + timeout

    while len(data) < size:
        if time.monotonic() >= deadline:
            raise BridgeError(
                f"Timed out while receiving binary data: {len(data):,}/{size:,} bytes."
            )

        remaining = size - len(data)
        chunk = stream.read(min(16384, remaining))

        if chunk:
            data.extend(chunk)
        else:
            time.sleep(0.001)

    return bytes(data)


def receive_vxf1_stream(
    stream: Union[serial.Serial, BinaryIO],
    frame_timeout: float = FRAME_TIMEOUT,
    progress_callback: bool = True,
) -> Tuple[bytes, int]:
    """
    Receive and validate continuous VXF1 binary frames until zero-length end frame.

    Returns:
        (accumulated_pcm_bytes, packet_count)
    """
    expected_sequence = 0
    pcm_buffer = bytearray()
    packet_count = 0
    last_reported_mb = 0.0

    while True:
        # 1. Read 16-byte frame header
        header_bytes = read_exact(stream, FRAME_HEADER_SIZE, frame_timeout)
        if len(header_bytes) != FRAME_HEADER_SIZE:
            raise BridgeError(
                f"Incomplete frame header: received {len(header_bytes)}/{FRAME_HEADER_SIZE} bytes."
            )

        magic, seq, payload_length, expected_crc = struct.unpack(
            FRAME_HEADER_FORMAT, header_bytes
        )

        # 2. Validate magic bytes
        if magic != FRAME_HEADER_MAGIC:
            raise BridgeError(
                f"Invalid frame magic: expected {FRAME_HEADER_MAGIC!r}, got {magic!r}"
            )

        # 3. Validate sequence number (detect loss, duplication, reordering)
        if seq != expected_sequence:
            raise BridgeError(
                f"Sequence mismatch: expected sequence {expected_sequence}, got {seq}"
            )

        # 4. Handle End-Of-Stream zero-length frame
        if payload_length == 0:
            if expected_crc != 0:
                raise BridgeError(
                    f"Zero-length final frame contained non-zero CRC32: {expected_crc:#010x}"
                )
            # Valid zero-length terminal frame reached
            break

        # 5. Validate payload constraints
        if payload_length > 65536 or payload_length % 2 != 0:
            raise BridgeError(
                f"Invalid frame payload length on sequence {seq}: {payload_length} bytes"
            )

        # 6. Read exact binary payload
        payload = read_exact(stream, payload_length, frame_timeout)
        if len(payload) != payload_length:
            raise BridgeError(
                f"Incomplete payload for frame {seq}: received {len(payload)}/{payload_length} bytes."
            )

        # 7. Validate CRC32
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise BridgeError(
                f"CRC32 mismatch on frame {seq}: expected {expected_crc:#010x}, calculated {actual_crc:#010x}"
            )

        # 8. Append validated PCM payload
        pcm_buffer.extend(payload)
        packet_count += 1
        expected_sequence += 1

        if progress_callback:
            curr_mb = len(pcm_buffer) / (1024 * 1024)
            if curr_mb - last_reported_mb >= 0.25:
                last_reported_mb = curr_mb
                dur = len(pcm_buffer) / (16000 * 2)
                print(
                    f"       Streaming: {len(pcm_buffer):,} bytes ({dur:5.2f}s, {packet_count} packets)"
                )

    return bytes(pcm_buffer), packet_count


def write_wav(pcm_bytes: bytes, sample_rate: int, output_path: Path) -> None:
    """Write signed 16-bit mono PCM as a valid WAV container."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def api_json(response: requests.Response, action: str) -> dict:
    """Validate an API response and return JSON."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise BridgeError(
            f"{action} failed: HTTP {response.status_code}: {response.text}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise BridgeError(f"{action} returned non-JSON data.") from exc


def make_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> io.BytesIO:
    """Create an in-memory WAV file for multipart upload."""
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    stream.seek(0)
    return stream


def run_bridge(
    port: str,
    baud: int,
    api_url: str,
    user_id: str,
    output_dir: Path,
    skip_api: bool,
    physical_buttons: bool = False,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"esp32_{timestamp}.wav"

    mode_label = (
        "Physical Buttons (GPIO27 START / GPIO33 STOP)"
        if physical_buttons
        else "Serial-Start"
    )

    print("=" * 64)
    print("VoxFlow ESP32 Real-Time Streaming Bridge (VXF1)")
    print(f"Mode     : {mode_label}")
    print(f"COM port : {port}")
    print(f"Baud     : {baud}")
    print(f"API      : {api_url}")
    print("=" * 64)

    session_id = None

    try:
        with serial.Serial(port=port, baudrate=baud, timeout=0.5) as ser:
            ser.dtr = False
            ser.rts = False
            # ESP32 commonly resets when the serial port is opened.
            # Wait past the firmware's startup delay before syncing.
            time.sleep(2.0)

            # Discard anything from the previous serial session.
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # ---------------------------------------------------------
            # Explicit handshake. This is more reliable than trusting
            # an old READY line left in the Windows serial buffer.
            # ---------------------------------------------------------
            print("[1/7] Synchronizing with ESP32...")
            sync_deadline = time.monotonic() + READLINE_TIMEOUT
            synced = False
            while time.monotonic() < sync_deadline:
                ser.write(b"PING\n")
                ser.flush()
                time.sleep(0.3)
                while ser.in_waiting > 0:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if line:
                        print(f"[ESP32] {line}")
                    if line == "PONG":
                        synced = True
                        break
                if synced:
                    break
            if not synced:
                raise BridgeError("Timed out waiting for 'PONG'.")

            # ---------------------------------------------------------
            # Create the VoxFlow session before capturing audio.
            # ---------------------------------------------------------
            if not skip_api:
                print("[2/7] Starting VoxFlow session...")
                base = api_url.rstrip("/")
                try:
                    start = requests.post(
                        f"{base}/api/v1/session/start",
                        json={"user_id": user_id},
                        timeout=15,
                    )
                except requests.RequestException as exc:
                    raise BridgeError(f"Cannot reach VoxFlow API at {base}: {exc}") from exc

                start_data = api_json(start, "Session start")
                try:
                    session_id = start_data["session_id"]
                except KeyError as exc:
                    raise BridgeError("Session start response has no session_id.") from exc

                print(f"       Session ID: {session_id}")
            else:
                print("[2/7] API upload skipped.")

            # ---------------------------------------------------------
            # ESP32 Capture initiation
            # ---------------------------------------------------------
            if physical_buttons:
                print("[3/7] Waiting for physical START button (GPIO27)...")
            else:
                print("[3/7] Sending START to ESP32...")
                ser.write(b"START\n")
                ser.flush()

            # Wait for RECORDING control header
            wait_for_exact_line(ser, "RECORDING", CAPTURE_HEADER_TIMEOUT)

            sample_rate_line = wait_for_prefix(ser, "SAMPLE_RATE=", READLINE_TIMEOUT)
            format_line = wait_for_prefix(ser, "FORMAT=", READLINE_TIMEOUT)
            stream_line = wait_for_prefix(ser, "STREAM=", READLINE_TIMEOUT)
            wait_for_exact_line(ser, "DATA_START", READLINE_TIMEOUT)

            try:
                sample_rate = int(sample_rate_line.split("=", 1)[1])
            except (ValueError, IndexError) as exc:
                raise BridgeError("Invalid ESP32 sample rate metadata.") from exc

            audio_format = format_line.split("=", 1)[1]
            stream_proto = stream_line.split("=", 1)[1]

            if audio_format != "PCM_S16LE_MONO":
                raise BridgeError(f"Unsupported ESP32 audio format: {audio_format}")

            if sample_rate != 16000:
                raise BridgeError(f"Unexpected sample rate from ESP32: {sample_rate}")

            if stream_proto != "VXF1":
                raise BridgeError(f"Unsupported stream protocol from ESP32: {stream_proto}")

            print(
                "[4/7] Receiving continuous audio stream (VXF1 framing @ 460800 baud)..."
            )

            # CRITICAL: Stream receiver parses binary frames directly without readline()
            pcm_bytes, packet_count = receive_vxf1_stream(ser, FRAME_TIMEOUT)

            # We are now at the text boundary after the final zero-length frame
            wait_for_exact_line(ser, "DONE", READLINE_TIMEOUT)

    except SerialException as exc:
        raise BridgeError(
            f"Could not open/use {port}. Make sure the ESP32 is connected, "
            f"the selected COM port is correct, and Arduino Serial Monitor is closed."
        ) from exc

    # -------------------------------------------------------------
    # Validate final audio stream constraints
    # -------------------------------------------------------------
    if len(pcm_bytes) % 2 != 0:
        raise BridgeError(f"Odd PCM byte count received: {len(pcm_bytes)} bytes.")

    duration = len(pcm_bytes) / (sample_rate * 2)
    if duration < 3.0 or duration > 60.0:
        raise BridgeError(
            f"Recording duration out of bounds: {duration:.2f}s "
            "(must be between 3.0s and 60.0s)"
        )

    # -------------------------------------------------------------
    # Local WAV is always produced, even when API upload is enabled.
    # -------------------------------------------------------------
    write_wav(pcm_bytes, sample_rate, output_path)

    print()
    print("=" * 64)
    print("STREAM RECONSTRUCTION SUMMARY")
    print(f"Packets received : {packet_count}")
    print(f"PCM bytes        : {len(pcm_bytes):,}")
    print(f"Duration         : {duration:.2f} sec")
    print("Sequence errors  : 0")
    print("CRC errors       : 0")
    print("Stream errors    : 0")
    print(f"WAV saved        : {output_path}")
    print("=" * 64)

    # -------------------------------------------------------------
    # Existing VoxFlow REST lifecycle: audio -> stop -> analyze.
    # -------------------------------------------------------------
    if not skip_api:
        assert session_id is not None
        base = api_url.rstrip("/")

        print("[6/7] Uploading WAV to VoxFlow...")
        wav_stream = make_wav_bytes(pcm_bytes, sample_rate)
        try:
            upload = requests.post(
                f"{base}/api/v1/session/{session_id}/audio",
                files={
                    "audio": (
                        output_path.name,
                        wav_stream,
                        "audio/wav",
                    )
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise BridgeError(f"Audio upload failed: {exc}") from exc
        finally:
            wav_stream.close()

        api_json(upload, "Audio upload")

        try:
            stop = requests.post(
                f"{base}/api/v1/session/{session_id}/stop",
                timeout=30,
            )
        except requests.RequestException as exc:
            raise BridgeError(f"Session stop failed: {exc}") from exc
        api_json(stop, "Session stop")

        print("[7/7] Running existing HuBERT analysis...")
        try:
            analyze = requests.post(
                f"{base}/api/v1/session/{session_id}/analyze",
                timeout=ANALYZE_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BridgeError(f"Session analysis request failed: {exc}") from exc

        result = api_json(analyze, "Session analysis")

        print()
        print("=" * 64)
        print("VOXFLOW SESSION COMPLETE")
        print(f"Session ID : {session_id}")
        print(f"Duration   : {result.get('duration', 'n/a')}")
        print(f"Windows    : {result.get('window_count', 'n/a')}")
        print(f"Events     : {result.get('event_count', 'n/a')}")
        print("=" * 64)

    else:
        print("API processing was skipped. WAV capture test completed successfully.")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture continuous speech sessions from VoxFlow ESP32 hardware."
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"ESP32 COM port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API,
        help=f"VoxFlow API base URL (default: {DEFAULT_API})",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER,
        help=f"VoxFlow user ID (default: {DEFAULT_USER})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory for captured WAV files.",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Only capture/save the WAV; do not call the VoxFlow API.",
    )
    parser.add_argument(
        "--physical-buttons",
        action="store_true",
        help="Wait for physical START button on ESP32 (GPIO27) instead of sending serial START.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        run_bridge(
            port=args.port,
            baud=args.baud,
            api_url=args.api_url,
            user_id=args.user_id,
            output_dir=args.output_dir,
            skip_api=args.skip_api,
            physical_buttons=args.physical_buttons,
        )
    except BridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


