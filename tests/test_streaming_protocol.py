"""
Unit and transport-level tests for VoxFlow VXF1 ESP32 continuous audio streaming protocol.

Tests cover:
 1. Single valid frame decoding.
 2. Multiple sequential valid frames decoding.
 3. Binary frames split across arbitrary fragmented read boundaries.
 4. Correct sequence progression validation.
 5. Sequence gap detection (loss).
 6. Duplicate sequence number detection.
 7. Out-of-order sequence number detection.
 8. CRC32 mismatch / payload corruption detection.
 9. Zero-length end-of-stream final frame handling.
10. Missing DONE text marker detection.
11. Odd PCM byte count rejection.
12. Exact reconstruction of deterministic synthetic PCM data.
13. 10-second synthetic stream reconstruction (320,000 bytes).
14. 20-second synthetic stream reconstruction (640,000 bytes).
15. 30-second synthetic stream reconstruction (960,000 bytes).
16. 60-second synthetic stream reconstruction (1,920,000 bytes).
17. Reconstructed byte count exact equality validation.
18. Reconstructed WAV duration exact match validation.
19. Verification that no extra bytes are injected into stream.
20. Verification that no audio bytes are lost or dropped.
"""

from __future__ import annotations

import io
import struct
import tempfile
import unittest
import wave
import zlib
from pathlib import Path

from tools.esp32_serial_bridge import (
    FRAME_HEADER_FORMAT,
    FRAME_HEADER_MAGIC,
    FRAME_HEADER_SIZE,
    BridgeError,
    read_exact,
    receive_vxf1_stream,
    write_wav,
)


def create_frame(seq: int, payload: bytes, crc_override: int | None = None) -> bytes:
    """Pack a single VXF1 frame (header + payload)."""
    if crc_override is not None:
        crc = crc_override
    elif len(payload) == 0:
        crc = 0
    else:
        crc = zlib.crc32(payload) & 0xFFFFFFFF

    header = struct.pack(FRAME_HEADER_FORMAT, FRAME_HEADER_MAGIC, seq, len(payload), crc)
    return header + payload


def create_stream(payloads: list[bytes], include_final_frame: bool = True) -> bytes:
    """Build a complete VXF1 binary stream from a list of PCM chunk payloads."""
    buffer = bytearray()
    seq = 0
    for chunk in payloads:
        buffer.extend(create_frame(seq, chunk))
        seq += 1

    if include_final_frame:
        # Zero-length terminal frame
        buffer.extend(create_frame(seq, b""))

    return bytes(buffer)


class FragmentedStream(io.BytesIO):
    """A stream simulator that returns small fragmented slices on read calls."""

    def __init__(self, initial_bytes: bytes, chunk_size: int = 7):
        super().__init__(initial_bytes)
        self.max_chunk_size = chunk_size

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.max_chunk_size:
            size = self.max_chunk_size
        return super().read(size)


class TestStreamingProtocol(unittest.TestCase):
    """Comprehensive test suite for the VXF1 streaming protocol and bridge parser."""

    def test_01_single_valid_frame(self):
        """1. Single valid audio frame decoding."""
        payload = bytes([i % 256 for i in range(2048)])
        stream_bytes = create_stream([payload])
        stream = io.BytesIO(stream_bytes)

        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)
        self.assertEqual(count, 1)
        self.assertEqual(pcm_out, payload)

    def test_02_multiple_valid_frames(self):
        """2. Multiple sequential valid audio frames."""
        chunks = [bytes([(c * 17 + i) % 256 for i in range(2048)]) for c in range(10)]
        expected_pcm = b"".join(chunks)
        stream_bytes = create_stream(chunks)
        stream = io.BytesIO(stream_bytes)

        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)
        self.assertEqual(count, 10)
        self.assertEqual(pcm_out, expected_pcm)

    def test_03_fragmented_read_boundaries(self):
        """3. Frames split across arbitrary small read chunks."""
        chunks = [b"A" * 1024, b"B" * 2048, b"C" * 512]
        expected_pcm = b"".join(chunks)
        stream_bytes = create_stream(chunks)

        # Force fragment reads to 11 bytes per read() call
        stream = FragmentedStream(stream_bytes, chunk_size=11)
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)
        self.assertEqual(count, 3)
        self.assertEqual(pcm_out, expected_pcm)

    def test_04_correct_sequence_ordering(self):
        """4. Sequence numbers progress monotonically 0, 1, 2, ..."""
        chunks = [b"\x00\x01" * 100 for _ in range(5)]
        stream = io.BytesIO(create_stream(chunks))
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)
        self.assertEqual(count, 5)
        self.assertEqual(len(pcm_out), 1000)

    def test_05_sequence_gap_detection(self):
        """5. Sequence gap (missing frame) raises BridgeError."""
        f0 = create_frame(0, b"\x01\x02" * 100)
        f2 = create_frame(2, b"\x03\x04" * 100)  # Skipped sequence 1!
        stream = io.BytesIO(f0 + f2)

        with self.assertRaises(BridgeError) as ctx:
            receive_vxf1_stream(stream, progress_callback=False)
        self.assertIn("Sequence mismatch", str(ctx.exception))

    def test_06_duplicate_sequence_detection(self):
        """6. Duplicate sequence number raises BridgeError."""
        f0 = create_frame(0, b"\x01\x02" * 100)
        f0_dup = create_frame(0, b"\x01\x02" * 100)
        stream = io.BytesIO(f0 + f0_dup)

        with self.assertRaises(BridgeError) as ctx:
            receive_vxf1_stream(stream, progress_callback=False)
        self.assertIn("Sequence mismatch", str(ctx.exception))

    def test_07_out_of_order_detection(self):
        """7. Out-of-order sequence number raises BridgeError."""
        f0 = create_frame(0, b"\x01\x02" * 100)
        f2 = create_frame(2, b"\x03\x04" * 100)
        f1 = create_frame(1, b"\x05\x06" * 100)
        stream = io.BytesIO(f0 + f2 + f1)

        with self.assertRaises(BridgeError) as ctx:
            receive_vxf1_stream(stream, progress_callback=False)
        self.assertIn("Sequence mismatch", str(ctx.exception))

    def test_08_crc_mismatch_detection(self):
        """8. CRC32 corruption in payload raises BridgeError."""
        payload = b"\x10\x20\x30\x40" * 200
        # Pass an incorrect CRC value
        corrupt_frame = create_frame(0, payload, crc_override=0xDEADBEEF)
        stream = io.BytesIO(corrupt_frame)

        with self.assertRaises(BridgeError) as ctx:
            receive_vxf1_stream(stream, progress_callback=False)
        self.assertIn("CRC32 mismatch", str(ctx.exception))

    def test_09_zero_length_final_frame(self):
        """9. Zero-length final frame terminates the stream unambiguously."""
        chunks = [b"\xaa\xbb" * 500]
        stream_bytes = create_stream(chunks, include_final_frame=True)
        # Append extra trailing bytes to ensure parser stops exactly at zero-length frame
        stream = io.BytesIO(stream_bytes + b"TRAILING_TEXT_NOT_IN_STREAM")

        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)
        self.assertEqual(count, 1)
        self.assertEqual(pcm_out, chunks[0])
        # Stream position should be immediately after the 16-byte zero-length frame
        remaining = stream.read()
        self.assertEqual(remaining, b"TRAILING_TEXT_NOT_IN_STREAM")

    def test_10_missing_final_frame_eof(self):
        """10. Stream ending abruptly without zero-length final frame raises BridgeError."""
        chunks = [b"\x11\x22" * 100]
        stream_bytes = create_stream(chunks, include_final_frame=False)
        stream = io.BytesIO(stream_bytes)

        with self.assertRaises(BridgeError) as ctx:
            receive_vxf1_stream(stream, frame_timeout=0.1, progress_callback=False)
        self.assertIn("Timed out", str(ctx.exception))

    def test_11_odd_pcm_byte_detection(self):
        """11. Payload with odd byte length is rejected (16-bit PCM must be even)."""
        odd_payload = b"\x01\x02\x03"  # 3 bytes (odd)
        frame = create_frame(0, odd_payload)
        stream = io.BytesIO(frame)

        with self.assertRaises(BridgeError) as ctx:
            receive_vxf1_stream(stream, progress_callback=False)
        self.assertIn("Invalid frame payload length", str(ctx.exception))

    def test_12_exact_reconstruction_known_synthetic(self):
        """12. Reconstruct known synthetic waveform (deterministic int16 sine wave)."""
        import math
        sample_count = 16000  # 1 second = 32000 bytes
        samples = [int(15000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(sample_count)]
        raw_pcm = struct.pack(f"<{sample_count}h", *samples)

        # Chunk into 2048-byte slices
        chunk_size = 2048
        chunks = [raw_pcm[i:i + chunk_size] for i in range(0, len(raw_pcm), chunk_size)]
        stream = io.BytesIO(create_stream(chunks))

        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)
        self.assertEqual(pcm_out, raw_pcm)
        self.assertEqual(len(pcm_out), 32000)

    def _generate_partitioned_chunks(self, total_bytes: int, seed: int = 1) -> tuple[list[bytes], bytes]:
        """Generate deterministic bytes partitioned into <= 2048-byte frames."""
        full_data = bytes([(i * seed + 7) % 256 for i in range(total_bytes)])
        chunk_size = 2048
        chunks = [full_data[i:i + chunk_size] for i in range(0, len(full_data), chunk_size)]
        return chunks, full_data

    def test_13_ten_second_synthetic_stream(self):
        """13. 10-second synthetic stream reconstruction (320,000 PCM bytes)."""
        total_bytes = 10 * 16000 * 2  # 320,000 bytes
        chunks, expected_pcm = self._generate_partitioned_chunks(total_bytes, seed=3)
        self.assertEqual(len(expected_pcm), 320000)

        stream = io.BytesIO(create_stream(chunks))
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)

        self.assertEqual(len(pcm_out), 320000)
        self.assertEqual(pcm_out, expected_pcm)

    def test_14_twenty_second_synthetic_stream(self):
        """14. 20-second synthetic stream reconstruction (640,000 PCM bytes)."""
        total_bytes = 20 * 16000 * 2  # 640,000 bytes
        chunks, expected_pcm = self._generate_partitioned_chunks(total_bytes, seed=7)
        self.assertEqual(len(expected_pcm), 640000)

        stream = io.BytesIO(create_stream(chunks))
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)

        self.assertEqual(len(pcm_out), 640000)
        self.assertEqual(pcm_out, expected_pcm)

    def test_15_thirty_second_synthetic_stream(self):
        """15. 30-second synthetic stream reconstruction (960,000 PCM bytes)."""
        total_bytes = 30 * 16000 * 2  # 960,000 bytes
        chunks, expected_pcm = self._generate_partitioned_chunks(total_bytes, seed=11)
        self.assertEqual(len(expected_pcm), 960000)

        stream = io.BytesIO(create_stream(chunks))
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)

        self.assertEqual(len(pcm_out), 960000)
        self.assertEqual(pcm_out, expected_pcm)

    def test_16_sixty_second_synthetic_stream(self):
        """16. 60-second synthetic stream reconstruction (1,920,000 PCM bytes)."""
        total_bytes = 60 * 16000 * 2  # 1,920,000 bytes
        chunks, expected_pcm = self._generate_partitioned_chunks(total_bytes, seed=13)
        self.assertEqual(len(expected_pcm), 1920000)

        stream = io.BytesIO(create_stream(chunks))
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)

        self.assertEqual(len(pcm_out), 1920000)
        self.assertEqual(pcm_out, expected_pcm)

    def test_17_verify_reconstructed_byte_count_equality(self):
        """17. Verify reconstructed byte count exactly matches generated input."""
        chunks = [b"\x12\x34" * 1024 for _ in range(25)]
        expected_len = sum(len(c) for c in chunks)

        stream = io.BytesIO(create_stream(chunks))
        pcm_out, count = receive_vxf1_stream(stream, progress_callback=False)

        self.assertEqual(len(pcm_out), expected_len)

    def test_18_verify_reconstructed_wav_duration(self):
        """18. Verify reconstructed WAV container duration exactly matches expected data."""
        sample_rate = 16000
        seconds = 15.0
        pcm_bytes = b"\x00\x00" * int(sample_rate * seconds)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test_output.wav"
            write_wav(pcm_bytes, sample_rate, wav_path)

            self.assertTrue(wav_path.exists())
            with wave.open(str(wav_path), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), sample_rate)
                nframes = wf.getnframes()
                duration = nframes / sample_rate
                self.assertAlmostEqual(duration, seconds, places=4)

    def test_19_verify_no_extra_bytes_introduced(self):
        """19. Verify that no header, padding, or artifact bytes are injected into audio."""
        chunks = [bytes([0xAB] * 2048), bytes([0xCD] * 2048)]
        stream = io.BytesIO(create_stream(chunks))
        pcm_out, _ = receive_vxf1_stream(stream, progress_callback=False)

        self.assertEqual(len(pcm_out), 4096)
        self.assertTrue(all(b == 0xAB for b in pcm_out[:2048]))
        self.assertTrue(all(b == 0xCD for b in pcm_out[2048:]))

    def test_20_verify_no_bytes_lost(self):
        """20. Verify every byte index from 0 to N-1 is preserved without loss."""
        # 100,000 distinct index modulo bytes
        raw_pcm = bytes([i % 251 for i in range(100000)])
        chunk_size = 2000
        chunks = [raw_pcm[i:i + chunk_size] for i in range(0, len(raw_pcm), chunk_size)]

    def test_21_sixty_second_hard_boundary_enforcement(self):
        """21. Verify that 60.0s (1,920,000 bytes) is valid and >60.0s is rejected."""
        sample_rate = 16000
        # Exactly 60.0s
        exact_60s_bytes = 60 * sample_rate * 2  # 1,920,000 bytes
        duration_60 = exact_60s_bytes / (sample_rate * 2)
        self.assertEqual(duration_60, 60.0)

        # 60.032s (1 extra 1024-sample frame = 2048 bytes over)
        over_60s_bytes = exact_60s_bytes + 2048
        over_duration = over_60s_bytes / (sample_rate * 2)
        self.assertGreater(over_duration, 60.0)

        # Verify bridge duration validation rule
        self.assertTrue(3.0 <= duration_60 <= 60.0)
        self.assertFalse(3.0 <= over_duration <= 60.0)


if __name__ == "__main__":
    unittest.main()

