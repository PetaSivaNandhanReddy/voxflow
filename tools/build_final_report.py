"""
build_full_report.py
Generates the pristine 31-page VoxFlow Final Project Report DOCX and PDF
in strict compliance with all faculty formatting and content requirements.
"""
import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=40, bottom=40, left=80, right=80):
    """Set inner padding for table cells in dxa."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    """Set background color of a cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="B0B0B0", sz="4", val="single"):
    """Set clean borders for table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_footer_page_number(run):
    """Inserts a dynamic Word PAGE field into a run."""
    fldSimple = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
    run._r.append(fldSimple)

def create_report():
    doc = Document()

    # Configure Default Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(2)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Page setup - A4, Margins: Left=1.25", Right=1.0", Top=1.0", Bottom=1.0"
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.0)
        
        # Configure footer page numbering (bottom-centre)
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        f_run = f_p.add_run()
        f_run.font.name = 'Times New Roman'
        f_run.font.size = Pt(10)
        add_footer_page_number(f_run)

    def add_title_p(text, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.first_line_indent = Inches(0)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        run.bold = bold
        return p

    def add_chapter_heading(text, page_break_before=False):
        p = doc.add_paragraph()
        if page_break_before:
            p.paragraph_format.page_break_before = True
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.line_spacing = 1.2
        run = p.add_run(text.upper())
        run.font.name = 'Times New Roman'
        run.font.size = Pt(16)
        run.bold = True
        return p

    def add_section_heading(text, page_break_before=False):
        p = doc.add_paragraph()
        if page_break_before:
            p.paragraph_format.page_break_before = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.line_spacing = 1.2
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.bold = True
        return p

    def add_subsection_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.line_spacing = 1.2
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.bold = True
        return p

    def add_body_p(text, indent=True, space_after=2):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(space_after)
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.4)
        else:
            p.paragraph_format.first_line_indent = Inches(0)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        return p

    def add_bullet_p(bold_prefix, text, space_after=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.2)
        
        run_bullet = p.add_run("• ")
        run_bullet.font.name = 'Times New Roman'
        run_bullet.font.size = Pt(12)
        run_bullet.bold = True
        
        if bold_prefix:
            run_b = p.add_run(bold_prefix + " ")
            run_b.font.name = 'Times New Roman'
            run_b.font.size = Pt(12)
            run_b.bold = True
            
        run_t = p.add_run(text)
        run_t.font.name = 'Times New Roman'
        run_t.font.size = Pt(12)
        return p

    def add_caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = False
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
        return p

    def add_figure_image(img_path, caption_text, width_inches=4.4, height_inches=None):
        if os.path.exists(img_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.first_line_indent = Inches(0)
            run = p.add_run()
            if height_inches:
                run.add_picture(img_path, height=Inches(height_inches))
            else:
                run.add_picture(img_path, width=Inches(width_inches))
            add_caption(caption_text)
        else:
            print(f"Warning: Image not found: {img_path}")

    def add_styled_table(headers, rows, col_widths=None):
        table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table)

        # Header Row
        hdr_cells = table.rows[0].cells
        for idx, title in enumerate(headers):
            hdr_cells[idx].text = title
            set_cell_shading(hdr_cells[idx], "E8EEF5")
            set_cell_margins(hdr_cells[idx], top=50, bottom=50, left=70, right=70)
            p = hdr_cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.1
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9)
                r.font.bold = True

        # Data Rows
        for r_idx, row_data in enumerate(rows):
            row_cells = table.rows[r_idx + 1].cells
            bg_color = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
            for c_idx, val in enumerate(row_data):
                row_cells[c_idx].text = str(val)
                if bg_color != "FFFFFF":
                    set_cell_shading(row_cells[c_idx], bg_color)
                set_cell_margins(row_cells[c_idx], top=40, bottom=40, left=70, right=70)
                p = row_cells[c_idx].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.line_spacing = 1.1
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.first_line_indent = Inches(0)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(8.5)

        # Column widths
        if col_widths:
            for row in table.rows:
                for idx, w in enumerate(col_widths):
                    row.cells[idx].width = Inches(w)

        return table

    def add_algorithm_box(algo_num, title, lines):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        cell.width = Inches(6.0)
        set_cell_shading(cell, "F8F9FA")
        set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
        
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="6" w:space="0" w:color="333333"/>
                <w:bottom w:val="single" w:sz="6" w:space="0" w:color="333333"/>
                <w:left w:val="single" w:sz="6" w:space="0" w:color="333333"/>
                <w:right w:val="single" w:sz="6" w:space="0" w:color="333333"/>
            </w:tcBorders>
        ''')
        tcPr.append(tcBorders)

        p0 = cell.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(2)
        p0.paragraph_format.line_spacing = 1.1
        p0.paragraph_format.first_line_indent = Inches(0)
        r0 = p0.add_run(f"Algorithm {algo_num}: {title}")
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(9.5)
        r0.font.bold = True

        for line in lines:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.first_line_indent = Inches(0)
            r = p.add_run(line)
            r.font.name = 'Consolas'
            r.font.size = Pt(8.0)

    def add_lit_entry(num_str, title_str, prob_str, meth_str, find_str, gap_str, rel_str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.first_line_indent = Inches(0)
        
        rh = p.add_run(f"{num_str} {title_str}\n")
        rh.font.name = 'Times New Roman'
        rh.font.size = Pt(11)
        rh.bold = True
        
        items = [
            ("Problem:", prob_str),
            ("Method:", meth_str),
            ("Key finding:", find_str),
            ("Limitation / gap:", gap_str),
            ("Relevance to VoxFlow:", rel_str)
        ]
        for label, text in items:
            rb = p.add_run(label + " ")
            rb.font.name = 'Times New Roman'
            rb.font.size = Pt(10)
            rb.bold = True
            
            rt = p.add_run(text + " ")
            rt.font.name = 'Times New Roman'
            rt.font.size = Pt(10)

    # =============================================================
    # PAGE 1: TITLE PAGE
    # =============================================================
    add_title_p("VOXFLOW — SPEECH FLUENCY & DISFLUENCY\nSESSION ANALYZER", size=18, bold=True, space_before=36, space_after=12)
    add_title_p("FINAL PROJECT REPORT", size=14, bold=True, space_after=8)
    add_title_p("Technical Answers for Real World Problems (TARP) — CSE1905", size=12, bold=False, space_after=28)
    
    add_title_p("Submitted by:", size=12, bold=True, space_after=6)
    add_title_p("23MIA1012 — Rayini Amarender Reddy\n23MIA1111 — Peta Siva Nandhan Reddy\n23MIA1140 — NayanUjwal Sri Tej", size=12, bold=False, space_after=28)
    
    add_title_p("Faculty Supervisor:", size=12, bold=True, space_after=6)
    add_title_p("Prof. GANALA SANTHOSHI", size=12, bold=True, space_after=28)
    
    add_title_p("Department of Computer Science and Engineering in Business Analytics\nSchool of Computer Science and Engineering (SCOPE)\nVellore Institute of Technology", size=12, bold=False, space_after=18)
    add_title_p("Project Repository: https://github.com/PetaSivaNandhanReddy/voxflow", size=11, bold=False, space_after=18)

    # =============================================================
    # PAGE 2: ABSTRACT
    # =============================================================
    add_chapter_heading("ABSTRACT", page_break_before=True)
    add_body_p("Speech disfluencies, such as repetitions, prolongations, and blocks, interrupt the smooth, natural rhythm of verbal expression. In clinical speech therapy, tracking therapeutic progress requires clinicians to listen to recorded sessions and manually tally disfluent events by hand. This manual counting is time-consuming, mentally demanding, and susceptible to inter-rater variability. While self-supervised speech representations have recently advanced automated stuttering detection, existing research focuses almost exclusively on classifying isolated 3-second audio clips without addressing the end-to-end engineering challenges of dedicated hardware capture, error-checked serial transport, continuous multi-window session inference, temporal event consolidation, and persistent historical tracking.")
    add_body_p("VoxFlow is an end-to-end speech fluency session analyzer designed to bridge this research gap. The system consists of an embedded hardware capture unit built on an ESP32 microcontroller and an INMP441 MEMS digital microphone recording 16 kHz, 16-bit mono PCM audio. Audio frames are streamed over USB-UART using a custom-engineered framed binary protocol (VXF1) equipped with 32-bit sequence tracking and CRC32 payload verification. A Python serial bridge reconstructs session WAV files and dispatches them to a Flask REST backend. The backend executes multi-label disfluency classification using HuBERT-D, a fine-tuned HuBERT Transformer evaluated on speaker-exclusive partitions of SEP-28k and FluencyBank (519 unique speakers). Temporal event aggregation consolidates overlapping 3-second window inferences into distinct, non-redundant clinical disfluency events. Session results, window probabilities, and aggregate metrics are stored in a local SQLite database and visualized via an interactive Streamlit dashboard. HuBERT-D achieves a test macro F1 of 0.4599 and a mean ROC-AUC of 0.8099 on unseen speakers, outperforming wav2vec 2.0 (0.4498) and an MFCC + SVM baseline (0.3121). All 38 automated unit and integration tests pass successfully, demonstrating a robust, fully verifiable session analysis platform.")

    # =============================================================
    # PAGE 3: CHAPTER 1: INTRODUCTION AND PROJECT OVERVIEW
    # =============================================================
    add_chapter_heading("1. INTRODUCTION AND PROJECT OVERVIEW", page_break_before=True)
    
    add_section_heading("1.1 Background")
    add_body_p("Stuttering is a neurodevelopmental speech disorder characterized by frequent disruptions in the forward flow of verbal expression. Clinically, disfluent speech patterns are categorized into three core behaviors: repetitions (involuntary iterations of sounds, syllables, or single-syllable words), prolongations (abnormal lengthening of continuous phonetic segments), and blocks (silent or audible postural fixations where airflow and vocalization are temporarily arrested). Stuttering affects approximately 1% of the global adult population and up to 5% of young children, impacting social communication, academic engagement, and emotional well-being.")
    add_body_p("Speech-language pathologists (SLPs) track therapeutic progress across multi-week interventions by evaluating disfluency frequency and severity during structured speech sessions. In standard clinical practice, therapists manually annotate audio or video recordings, tallying individual disfluency occurrences to compute standardized metrics such as the Percentage of Stuttered Syllables (%SS) and Stuttering Severity Instrument (SSI) scores. However, manual perceptual evaluation is labor-intensive, exhausting, and prone to substantial subjective disagreement across raters [1]. Recent advances in computational speech processing have introduced machine learning models for automated disfluency detection using publicly available corpora such as SEP-28k [2] and FluencyBank [3]. Nevertheless, the vast majority of existing academic literature remains confined to classifying isolated 3-second audio clips, leaving an unaddressed engineering gap between algorithmic clip classification and continuous, hardware-integrated clinical session analysis.")

    add_section_heading("1.2 Project Overview")
    add_body_p("VoxFlow addresses this challenge by providing a complete, automated speech fluency and disfluency session analyzer. The system operates through a structured end-to-end pipeline:")
    add_bullet_p("Record session:", "The user initiates recording using a dedicated ESP32 + INMP441 physical capture unit (3 to 60 seconds).")
    add_bullet_p("Process audio:", "Audio is streamed over serial via the verified VXF1 protocol, reconstructed into a standardized 16 kHz mono WAV file, validated, and sliced into 3.0-second sliding analysis windows with a 1.0-second hop.")
    add_bullet_p("Detect disfluencies:", "HuBERT-D performs multi-label inference on each window, generating independent probabilities for repetition, prolongation, and block classes.")
    add_bullet_p("Aggregate events:", "A temporal event aggregation algorithm merges contiguous and proximate window activations (within a 1.5-second merge gap) into unified, non-redundant clinical events.")
    add_bullet_p("Store results:", "Session metadata, per-window probability vectors, and aggregated disfluency events are persisted to a relational SQLite database.")
    add_bullet_p("Review history:", "An interactive Streamlit visual analytics dashboard displays session timelines, per-class breakdown charts, and longitudinal progress trends across multiple sessions.")

    add_section_heading("1.3 Project Objectives")
    add_body_p("The core objectives of the VoxFlow project are defined as follows:")
    add_bullet_p("Objective 1 (Hardware Capture & Transport):", "Design and construct a low-cost, dedicated digital speech capture unit using an ESP32 microcontroller and INMP441 MEMS microphone, with an error-checked framed binary streaming protocol (VXF1) ensuring zero silent data loss.")
    add_bullet_p("Objective 2 (Continuous Speech Analysis Engine):", "Implement a backend processing pipeline that validates session audio (3–60 s), handles arbitrary session lengths via sliding windows (3.0 s window, 1.0 s hop), and reliably classifies core disfluencies.")
    add_bullet_p("Objective 3 (Self-Supervised Disfluency Modeling):", "Fine-tune a self-supervised transformer model (HuBERT-D) on speaker-exclusive partitions of SEP-28k and FluencyBank, rigorously evaluating performance against wav2vec 2.0 and traditional acoustic baselines.")
    add_bullet_p("Objective 4 (Temporal Consolidation & Visual Analytics):", "Develop a temporal event aggregator to prevent multi-counting across overlapping analysis windows, backed by relational database persistence and an intuitive clinical dashboard for session review and longitudinal trend analysis.")

    # =============================================================
    # PAGE 4: CHAPTER 2: PROBLEM STATEMENT, RESEARCH GAP AND NOVELTY
    # =============================================================
    add_chapter_heading("2. PROBLEM STATEMENT, RESEARCH GAP AND NOVELTY", page_break_before=False)
    
    add_section_heading("2.1 Problem Statement")
    add_body_p("Tracking speech fluency over longitudinal therapeutic interventions requires repeated speech assessments. In current clinical and therapeutic workflows, clinicians and speech therapists must manually listen to recorded sessions and hand-count individual repetitions, prolongations, and blocks. This manual observation demands substantial clinical time, imposes a heavy cognitive workload, and complicates objective comparison of patient progress across consecutive weeks.")
    add_body_p("When automated speech processing is introduced to assist this workflow, sliding analysis windows applied across continuous speech naturally produce overlapping detections: a single 2-second block or prolongation spanning across multiple consecutive windows is counted multiple times if left unmerged. Furthermore, therapists require structured session-level information—including total session duration, speaking time, total disfluency count, disfluency rate per minute, and chronological event timelines—rather than isolated clip labels. Therefore, the core engineering problem is: how can we capture speech sessions with low-cost hardware, detect repetitions, prolongations, and blocks across continuous speech without redundant window multi-counting, and store structured session results so that therapeutic progress can be objectively reviewed over time? VoxFlow is designed as an assistive research tool to provide structured measurements and does not replace professional clinical diagnosis.")

    add_section_heading("2.2 Existing Limitations")
    add_body_p("A rigorous examination of existing academic literature and software tools reveals several recurring limitations:")
    add_bullet_p("Clip-Level Isolation:", "Nearly all published disfluency detectors operate exclusively on isolated 3-second audio segments pre-trimmed from podcasts or clinical archives, failing to evaluate continuous conversational speech.")
    add_bullet_p("Multi-Counting in Sliding Windows:", "Applying clip classifiers across continuous sessions using sliding windows causes individual disfluencies to trigger in several consecutive windows, falsely inflating event counts without temporal consolidation.")
    add_bullet_p("Overoptimistic Splitting Schemes:", "Many studies utilize random train/test splits that leak speaker identities across sets, producing inflated benchmark scores that fail to generalize to unseen speakers.")
    add_bullet_p("Absence of Integrated Capture Hardware:", "Existing studies focus on offline machine learning notebooks and rarely specify the hardware capture chain, microphone electrical interfacing, or transmission error handling.")
    add_bullet_p("Lack of Longitudinal Storage & Visualization:", "Academic prototypes rarely incorporate persistent database storage or clinical dashboard interfaces for tracking fluency trends across multiple sessions.")

    add_section_heading("2.3 Research Gap")
    add_body_p("In the set of published studies reviewed (detailed in Chapter 3), we did not find a unified system that integrates: (a) dedicated low-cost embedded speech capture with hardware-level transport verification, (b) continuous multi-window self-supervised disfluency detection evaluated under speaker-exclusive constraints, (c) temporal event aggregation to eliminate overlapping window redundancy, and (d) relational session persistence with longitudinal visual analytics. Existing work addresses isolated sub-problems—such as model architectures or dataset curation—leaving an end-to-end systems gap for session-level fluency tracking.")

    add_section_heading("2.4 Novelty and Proposed Contribution")
    add_body_p("VoxFlow does not claim to introduce a novel neural architecture from scratch; rather, its novelty and contribution reside in its comprehensive, verified system-level integration:")
    add_bullet_p("Framed Hardware Streaming (VXF1):", "Engineered an embedded binary protocol on ESP32 that embeds frame magic headers, 32-bit sequence numbers, payload lengths, and CRC32 checksums, ensuring transmission integrity across USB-UART.")
    add_bullet_p("Speaker-Exclusive HuBERT-D Modeling:", "Fine-tuned HuBERT-D on 30,999 multi-annotator audio clips across 519 strictly partitioned, speaker-exclusive individuals, ensuring realistic generalization on unseen voices.")
    add_bullet_p("Temporal Event Aggregation Algorithm:", "Formulated an algorithmic consolidation mechanism that clusters overlapping window detections within a 1.5-second gap into single, bounded clinical events.")
    add_bullet_p("Complete Dual-Mode Software Pipeline:", "Developed an integrated platform supporting both live hardware recording and standalone WAV file uploads, with relational SQLite persistence and Streamlit visual analytics.")

    # =============================================================
    # PAGE 5 & 6: CHAPTER 3: LITERATURE REVIEW
    # =============================================================
    add_chapter_heading("3. LITERATURE REVIEW", page_break_before=False)
    
    add_section_heading("3.1 Reviewed Literature")
    add_body_p("To establish a rigorous theoretical and empirical foundation for VoxFlow, we conducted a systematic literature survey of 13 key peer-reviewed research papers spanning speech disfluency datasets, acoustic modeling, deep learning architectures, and self-supervised speech representations.")

    add_lit_entry("3.1.1", "Sheikh et al. (2022) [1]",
                  "Review machine learning techniques for stuttering identification, covering features, classifiers, datasets, and practical challenges.",
                  "Systematic survey of acoustic features (MFCCs, spectrograms), classical classifiers (SVMs, GMMs), and deep neural architectures.",
                  "Identified dataset fragmentation, lack of standardized evaluation protocols, and dominance of short-clip classification over continuous sessions.",
                  "Survey paper; did not construct an end-to-end hardware capture, session aggregation, or clinical software tool.",
                  "Validates VoxFlow's system-level objective to transition from isolated clip detection to full session tracking.")

    add_lit_entry("3.1.2", "Lea et al. (2021) [2]",
                  "Scarcity of large-scale, naturalistic, publicly available audio datasets for multi-class stuttering event detection.",
                  "Introduced the SEP-28k dataset (28,177 3-second clips from public podcasts) annotated for repetitions, prolongations, blocks, interjections, and word repetitions.",
                  "Demonstrated multi-label disfluency detection in conversational audio is challenging, establishing baseline classifier benchmarks.",
                  "Dataset audio is segmented into isolated 3-second clips without continuous temporal context or hardware streaming.",
                  "SEP-28k serves as the primary training and evaluation corpus for VoxFlow's HuBERT-D classifier.")

    add_lit_entry("3.1.3", "Bernstein Ratner and MacWhinney (2018) [3]",
                  "Absence of shared, open-access clinical archives and standardized transcripts for stuttering and speech fluency research.",
                  "Established FluencyBank under the TalkBank framework, providing standardized audio recordings and clinical transcripts.",
                  "Created an international open-access resource that enabled reproducible computational and clinical speech research.",
                  "Archival database; does not provide automated machine learning detection models or hardware capture utilities.",
                  "FluencyBank provides the secondary benchmark corpus incorporated into VoxFlow's multi-annotator training dataset.")

    add_lit_entry("3.1.4", "Howell et al. (2009) [4]",
                  "Need for accessible clinical recordings to analyze developmental stuttering trajectories across age cohorts.",
                  "Developed the University College London Archive of Stuttered Speech (UCLASS), offering recordings and orthographic transcriptions.",
                  "Showed acoustic differences in disfluency manifestations across age groups, establishing an early public benchmark.",
                  "Smaller dataset recorded under legacy acoustic conditions; evaluated on isolated speech tokens without modern neural models.",
                  "Provides foundational clinical taxonomy for defining repetition, prolongation, and block categories.")

    add_lit_entry("3.1.5", "Kourkounakis et al. (2021) [5]",
                  "Detecting multi-class stuttered disfluencies directly from spectrogram representations without manual feature engineering.",
                  "Proposed FluentNet, integrating Squeeze-and-Excitation ResNet, BiLSTM layers, and attention mechanisms on log-mel spectrograms.",
                  "Achieved 91.75% accuracy on UCLASS and 86.7% on LibriStutter, proving attention mechanisms capture disfluency boundaries.",
                  "Evaluated strictly on pre-cut audio segments; did not address streaming capture, frame transport, or session aggregation.",
                  "Confirms the importance of capturing temporal dependencies in disfluency classification, inspiring VoxFlow's windowing design.")

    add_lit_entry("3.1.6", "Sheikh et al. (2021) [6]",
                  "High computational complexity and inference latency of recurrent neural networks in multi-class stuttering classification.",
                  "Developed StutterNet, a Time-Delay Neural Network (TDNN) architecture trained on 2D MFCC acoustic features across context frames.",
                  "Outperformed baseline residual architectures while significantly lowering parameter counts through sub-sampled convolutions.",
                  "Operates on isolated clips with handcrafted MFCC features, which have lower representational capacity than self-supervised embeddings.",
                  "Demonstrates the feasibility of low-latency frame-level disfluency scoring and provides a baseline comparison.")

    add_lit_entry("3.1.7", "Sheikh et al. (2023) [7]",
                  "Severe class imbalance across disfluency types and poor generalization across unseen speaker distributions in SEP-28k.",
                  "Proposed multi-contextual deep learning using SpecAugment data augmentation, class-balanced focal loss, and multi-branch features.",
                  "Class-balanced loss functions and data augmentation significantly improved detection of minority disfluencies (blocks and prolongations).",
                  "Clip-level evaluation without physical hardware integration, serial error checking, or session-level event consolidation.",
                  "Directly motivated VoxFlow's adoption of class-weighted loss and disfluency-specific detection thresholds.")

    add_lit_entry("3.1.8", "Bayerl et al. (2022a) [8]",
                  "Scarcity of clinical therapy datasets capturing deliberate fluency-shaping techniques alongside natural disfluencies.",
                  "Constructed the Kassel State of Fluency (KSoF) dataset, consisting of German speech recordings annotated by professional speech therapists.",
                  "Showed that clinical evaluation requires fine-grained disfluency categorization rather than binary fluent/disfluent discrimination.",
                  "Language-specific (German) corpus focused on therapy modifications; evaluated on isolated clips without an integrated pipeline.",
                  "Underlines the clinical necessity of independently tracking repetition, prolongation, and block classes across assessment sessions.")

    add_lit_entry("3.1.9", "Bayerl et al. (2022b) [9]",
                  "Suboptimal classification performance of conventional handcrafted acoustic features (MFCCs, filterbanks) on subtle disfluency cues.",
                  "Extracted deep acoustic representations from fine-tuned wav2vec 2.0 self-supervised models and evaluated them with SVM classifiers.",
                  "Fine-tuned self-supervised speech representations achieved relative F1 score improvements of up to 27% over traditional baselines.",
                  "Relied on offline feature extraction and clip-level classifiers without session streaming or persistent database storage.",
                  "Provides foundational empirical justification for using pre-trained self-supervised transformer backbones in VoxFlow.")

    add_lit_entry("3.1.10", "Bayerl et al. (2022c) [10]",
                  "Overoptimistic performance claims in stuttering literature caused by speaker leakage across training and test splits.",
                  "Extended SEP-28k with verified speaker identity labels (SEP-28k-Extended) and compared random clip splitting vs. speaker-exclusive partitioning.",
                  "Proved random splitting severely overestimates detection accuracy; speaker-exclusive splits provide the only valid benchmark for real-world generalization.",
                  "Focused on dataset partitioning methodology without implementing capture hardware or session-level software pipelines.",
                  "VoxFlow directly adopts speaker-exclusive dataset partitioning (519 unique speakers, 0 overlap) to guarantee rigorous validation.")

    add_lit_entry("3.1.11", "Bayerl et al. (2023) [11]",
                  "Need for standardized comparative benchmarking of diverse acoustic, phonetic, and neural feature sets for stuttering classification.",
                  "Analyzed results from the Interspeech ComParE 2022 Stuttering Sub-Challenge, benchmarking standard acoustic sets vs. deep embeddings.",
                  "Pre-trained deep acoustic representations consistently and significantly outperformed handcrafted feature engineering.",
                  "Focused exclusively on comparative challenge metrics without addressing real-time capture, serial framing, or clinical session interfaces.",
                  "Supports VoxFlow's architectural choice of fine-tuned self-supervised transformers over baseline MFCC pipelines.")

    add_lit_entry("3.1.12", "Baevski et al. (2020) [12]",
                  "Training high-performing speech recognition models requires thousands of hours of scarce, expensive human-annotated speech.",
                  "Introduced wav2vec 2.0, a self-supervised learning framework that masks latent speech representations and optimizes a contrastive task.",
                  "Self-supervised pre-training on unannotated audio builds rich acoustic representations that transfer effectively to downstream speech tasks.",
                  "General speech foundation model; requires fine-tuning and task-specific architecture for multi-label disfluency classification.",
                  "wav2vec 2.0 serves as the primary comparative self-supervised baseline in VoxFlow's model benchmarking.")

    add_lit_entry("3.1.13", "Hsu et al. (2021) [13]",
                  "Contrastive speech models face optimization instability and risk conflating acoustic background noise with phonetic units.",
                  "Proposed HuBERT (Hidden-Unit BERT), using offline k-means clustering to create discrete targets followed by masked language modeling on continuous audio.",
                  "HuBERT achieves superior acoustic and phonetic representation stability and out-of-domain generalization compared to contrastive frameworks.",
                  "Foundation model trained on fluent read speech; requires multi-label fine-tuning and threshold calibration for disordered speech.",
                  "HuBERT serves as the foundational backbone for VoxFlow's fine-tuned HuBERT-D classifier (Macro F1 = 0.4599, ROC-AUC = 0.8099).")

    add_section_heading("3.2 Comparison of Existing Approaches")
    add_body_p("A comparative synthesis of existing approaches highlights a clear progression in the field: early foundational efforts (such as UCLASS [4] and FluencyBank [3]) established shared clinical corpora, while subsequent research (such as FluentNet [5] and StutterNet [6]) proved that deep neural networks operating on spectrograms and MFCCs could detect disfluencies automatically. More recently, self-supervised foundation models (wav2vec 2.0 [12] and HuBERT [13]) fine-tuned on stuttering datasets (Bayerl et al. [9], [11]) significantly boosted classification accuracy over handcrafted features.")
    add_body_p("However, across all reviewed literature, existing systems operate almost exclusively on isolated, pre-trimmed 3-second audio clips. None of the reviewed approaches combine physical microphone capture, error-checked serial transport, sliding-window session inference, temporal event consolidation (to prevent multi-counting across overlapping windows), and relational session persistence in a single deployable software-hardware framework. VoxFlow directly bridges this gap.")

    add_section_heading("3.3 Research Gap from Literature")
    add_body_p("The synthesis of reviewed literature reveals three fundamental gaps:")
    add_bullet_p("System-Level Integration Gap:", "Academic studies focus heavily on machine learning model architectures evaluated in Jupyter notebooks, leaving the physical audio capture, real-time transport, and user interface unaddressed.")
    add_bullet_p("Continuous Session Processing Gap:", "Classifiers designed for 3-second clips cannot be naively deployed on continuous recordings without multi-window slicing and temporal event aggregation to merge redundant detections.")
    add_bullet_p("Longitudinal Tracking Gap:", "Clinicians require structured historical tracking across successive sessions to evaluate intervention efficacy, which existing clip-level benchmarks do not support.")
    add_body_p("VoxFlow resolves these gaps by integrating low-cost ESP32 hardware streaming, the error-checked VXF1 protocol, speaker-exclusive HuBERT-D modeling, temporal event aggregation, and SQLite-backed Streamlit visual analytics.")

    # =============================================================
    # PAGE 7: CHAPTER 4: FEASIBILITY STUDY
    # =============================================================
    add_chapter_heading("4. FEASIBILITY STUDY", page_break_before=False)
    
    add_section_heading("4.1 Technical Feasibility")
    add_body_p("The technical feasibility of VoxFlow was verified by implementing and testing every subsystem across the complete pipeline: ESP32 embedded firmware, Python serial bridge, Flask REST API, HuBERT-D inference engine, SQLite database, and Streamlit dashboard. The backend runs on standard consumer PC hardware, utilizing CUDA GPU acceleration when available and seamlessly falling back to multi-threaded CPU execution. All software dependencies are open-source and cross-platform.")

    add_section_heading("4.2 Hardware Feasibility")
    add_body_p("The hardware architecture utilizes an ESP32 Dev Module (ESP-WROOM-32) paired with an INMP441 MEMS digital microphone. The INMP441 provides direct I2S digital output (16 kHz, 16-bit mono), completely eliminating the need for an external analogue-to-digital converter (ADC) and avoiding analog signal degradation. Both components are low-cost, readily available, and operate reliably over standard USB power. At 16 kHz mono PCM, audio generates 32,000 bytes per second. Over a 460,800 baud serial connection (effective throughput ~46,080 bytes/s), the USB-UART interface provides ~44% excess bandwidth, guaranteeing smooth, unbuffered real-time streaming.")

    add_section_heading("4.3 Software Feasibility")
    add_body_p("The software architecture is built on Python 3.11+ using proven, industry-standard frameworks: Flask for RESTful service endpoints, PyTorch and Hugging Face Transformers for HuBERT-D model execution, SQLite3 for local zero-configuration relational persistence, and Streamlit for rapid, interactive visual analytics. Automated testing is managed via Python's standard unittest framework.")

    add_section_heading("4.4 Dataset and Data Feasibility")
    add_body_p("Model development is based on the SEP-28k [2] and FluencyBank [3] datasets. To ensure clinical validity and eliminate label noise, the raw dataset was rigorously audited. Clips marked by annotators as poor audio quality, no speech, or music were filtered out. Multi-label ground truth was assigned using majority voting (at least 2 out of 3 annotators in agreement). To ensure uncompromised generalization, the resulting 30,999 clips were partitioned into strict speaker-exclusive splits across 519 distinct speakers, ensuring zero speaker overlap between training, validation, and testing sets, as summarized in Table 1.")

    add_caption("Table 1. Dataset Summary and Speaker-Exclusive Splits")
    add_styled_table(
        ["Split", "Clips", "Speakers", "Repetition", "Prolongation", "Block", "Multi-Label"],
        [
            ["Train", "17,555", "21", "3,242", "1,702", "1,984", "630"],
            ["Validation", "6,632", "249", "1,279", "652", "845", "280"],
            ["Test", "6,812", "249", "1,190", "694", "854", "262"],
            ["Total", "30,999", "519", "5,711", "3,048", "3,683", "1,172"]
        ],
        col_widths=[1.0, 0.8, 0.8, 0.9, 0.9, 0.8, 0.8]
    )

    add_section_heading("4.5 Integration Feasibility")
    add_body_p("Subsystem integration is decoupled through clean, standardized interfaces: the ESP32 communicates with the PC bridge via binary VXF1 frames; the bridge writes standard WAV files and invokes the backend via HTTP REST endpoints; the backend interacts with the SQLite database via structured SQL schemas; and the Streamlit dashboard queries REST endpoints and database records. This modular decoupling allows individual components to be updated or tested independently.")

    add_section_heading("4.6 Field Work and Data Collection")
    add_body_p("In adherence to academic integrity, we explicitly declare that no formal clinical field visits, hospital trials, patient interviews, or clinician questionnaires were conducted for this prototype. Data collection and empirical validation consisted of: (1) downloading, auditing, and structuring public benchmark speech datasets (SEP-28k and FluencyBank) into speaker-exclusive manifests, and (2) laboratory-based audio recording and hardware transmission tests using the assembled ESP32 + INMP441 capture unit.")

    add_section_heading("4.7 Planned Modules")
    add_body_p("To ensure a structured, verifiable engineering lifecycle, the VoxFlow system was designed and implemented across four planned core modules:")
    add_bullet_p("Module 1 — Hardware Capture Unit:", "ESP32 Dev Module and INMP441 digital MEMS microphone with FreeRTOS dual-core task scheduling, capturing 16 kHz 16-bit mono audio and encapsulating it into VXF1 frames.")
    add_bullet_p("Module 2 — Serial Bridge & WAV Reconstructor:", "Python serial receiver that synchronizes frame headers, verifies CRC32 checksums, detects missing packets via sequence numbers, and writes monolithic session WAV files.")
    add_bullet_p("Module 3 — Backend API & AI Inference Engine:", "Flask REST backend executing audio validation, 3.0-second sliding-window generation with 1.0-second hop, multi-label HuBERT-D inference, and temporal event aggregation.")
    add_bullet_p("Module 4 — Clinical Visual Analytics Dashboard:", "Streamlit-based user interface enabling real-time session capture, audio upload, timeline inspection, disfluency metrics display, and longitudinal history review.")
    add_body_p("All four planned modules were fully implemented, integrated, and verified in the final codebase.")

    # =============================================================
    # PAGE 8 & 9: CHAPTER 5: SYSTEM REQUIREMENTS
    # =============================================================
    add_chapter_heading("5. SYSTEM REQUIREMENTS", page_break_before=False)
    
    add_section_heading("5.1 Hardware Requirements")
    add_body_p("The hardware components required for the VoxFlow session capture system are detailed in Table 2.")
    
    add_caption("Table 2. Hardware Components and Specifications")
    add_styled_table(
        ["Component", "Purpose", "Key Specifications"],
        [
            ["ESP32 Dev Module (ESP-WROOM-32)", "Capture controller & streaming unit", "Dual-core Xtensa LX6 @ 240 MHz, 520 KB SRAM, hardware I2S, USB-UART"],
            ["INMP441 MEMS Microphone", "Acoustic speech capture", "Omnidirectional digital microphone, I2S 24-bit/16-bit output, SNR 61 dBA"],
            ["Push Buttons (2x)", "Hardware session control", "START (GPIO27) and STOP (GPIO33), active LOW with internal pull-ups"],
            ["Status LED", "Visual operational feedback", "GPIO4 with 330 Ohm resistor; solid on standby, 2 Hz blinking during recording"],
            ["Micro-USB Cable & Breadboard", "Power & serial interface", "Standard USB 2.0 cable providing 5V power and 460,800 baud data link"],
            ["Host PC / Laptop", "Processing, AI & visualization", "x86_64 or ARM64 PC, 8 GB+ RAM, Python 3.11+, optional CUDA GPU"]
        ],
        col_widths=[1.8, 1.8, 2.4]
    )

    add_section_heading("5.2 Software Requirements")
    add_body_p("The software dependencies, runtime environments, and core libraries are listed in Table 3.")
    
    add_caption("Table 3. Software Requirements and Dependencies")
    add_styled_table(
        ["Software / Tool", "Version", "Role / Usage"],
        [
            ["Arduino IDE / CLI", "2.x", "Firmware compilation and flashing to ESP32"],
            ["ESP32 Arduino Core", "2.0.14+", "ESP32 board support, FreeRTOS and I2S driver"],
            ["Python", "3.11+", "Primary programming language for bridge, backend, and UI"],
            ["PyTorch", "2.0+", "Deep learning runtime for HuBERT-D inference"],
            ["Hugging Face Transformers", "4.38+", "HuBERT model loading and feature extraction"],
            ["Flask", "3.0+", "REST API backend framework"],
            ["Streamlit", "1.30+", "Web-based interactive visual analytics dashboard"],
            ["SQLite3", "3.x", "Embedded zero-configuration relational database"],
            ["PySerial & SoundFile", "3.5 / 0.12", "Serial communication and WAV audio encoding/decoding"]
        ],
        col_widths=[1.8, 1.0, 3.2]
    )

    add_section_heading("5.3 Functional Requirements", page_break_before=False)
    add_body_p("The functional requirements governing system behavior are enumerated in Table 4.")
    
    add_caption("Table 4. Functional Requirements (FR)")
    add_styled_table(
        ["Requirement ID", "Functional Requirement Description"],
        [
            ["FR-01: Hardware Capture", "The system shall capture 16 kHz, 16-bit mono PCM speech via I2S and stream framed VXF1 packets over serial."],
            ["FR-02: Session Limits", "The system shall support speech session recordings between 3.0 seconds (minimum) and 60.0 seconds (maximum)."],
            ["FR-03: Frame Verification", "The serial bridge shall verify frame magic headers, consecutive sequence numbers, and CRC32 checksums."],
            ["FR-04: WAV Reconstruction", "The bridge shall reconstruct received frames into a valid 16 kHz mono WAV file upon session completion."],
            ["FR-05: Windowing & AI Inference", "The backend shall segment session audio into 3.0 s windows (1.0 s hop) and predict disfluency probabilities via HuBERT-D."],
            ["FR-06: Event Aggregation", "The backend shall aggregate consecutive/proximate window detections (gap <= 1.5 s) into unified clinical events."],
            ["FR-07: Relational Persistence", "The database shall store complete session records, per-window predictions, and aggregated events."],
            ["FR-08: Dashboard Visualization", "The dashboard shall display session timelines, per-class event tallies, disfluency rates, and historical trends."]
        ],
        col_widths=[1.5, 4.5]
    )

    add_section_heading("5.4 Non-Functional Requirements")
    add_body_p("The non-functional performance, reliability, and security requirements are summarized in Table 5.")
    
    add_caption("Table 5. Non-Functional Requirements (NFR)")
    add_styled_table(
        ["Requirement ID", "Non-Functional Requirement Description"],
        [
            ["NFR-01: Low Latency", "End-to-end session analysis (validation, inference, aggregation, database write) shall complete in < 5.0 s for a 60 s recording on GPU."],
            ["NFR-02: Data Integrity", "Zero undetected frame corruption: all corrupted or dropped serial frames shall be flagged via CRC32 and sequence checks."],
            ["NFR-03: Portability", "The software backend shall run locally on Windows, Linux, and macOS without mandatory cloud connectivity."],
            ["NFR-04: Privacy & Security", "All patient audio recordings, transcripts, and analysis metrics shall remain stored locally on the host machine."],
            ["NFR-05: Modularity", "Subsystems (bridge, API, model, UI) shall remain decoupled, communicating exclusively via standard protocols (VXF1, REST, SQL)."],
            ["NFR-06: Robustness", "The API shall gracefully reject invalid audio (silence, non-16kHz, out-of-bounds duration) returning descriptive HTTP 422 errors."]
        ],
        col_widths=[1.5, 4.5]
    )

    # =============================================================
    # PAGE 10: CHAPTER 6: SYSTEM DESIGN (6.1 & Fig. 1)
    # =============================================================
    add_chapter_heading("6. SYSTEM DESIGN", page_break_before=False)
    
    add_section_heading("6.1 Overall Architecture Design")
    add_body_p("The VoxFlow system follows a modular, four-tier architecture spanning embedded hardware capture, serial communication bridging, REST backend processing, and visual analytics presentation, as illustrated in Fig. 1. Audio originates at the INMP441 MEMS microphone, where sound waves are digitized into 16 kHz 16-bit mono PCM. The ESP32 captures audio via I2S, encapsulates it into VXF1 frames, and transmits it over USB-UART to the host PC. The Python serial bridge receives frames, validates CRC32 checksums, and reconstructs a single monolithic WAV file. The Flask REST backend manages audio validation, sliding-window generation, HuBERT-D multi-label inference, and temporal event aggregation. Session metadata and results are stored in SQLite and rendered on the Streamlit dashboard.")

    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\architecture_design\08_component.png",
        "Fig. 1. Overall Component Architecture.",
        width_inches=3.2
    )

    # =============================================================
    # PAGE 11: 6.2 UML Design & Fig. 2 Use Case
    # =============================================================
    add_section_heading("6.2 UML Design", page_break_before=False)
    add_body_p("To formally model system behavior, interactions, and structural relationships, a comprehensive suite of Unified Modeling Language (UML) diagrams was developed.")
    add_body_p("The Use Case Diagram (Fig. 2) illustrates the primary user interactions supported by VoxFlow: starting/stopping hardware recording, uploading pre-recorded WAV sessions, viewing session analysis timelines, and reviewing longitudinal disfluency trends.")
    
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\01_use_case.png",
        "Fig. 2. Use Case Diagram.",
        width_inches=3.6
    )

    # =============================================================
    # PAGE 12: Fig. 3 Class Diagram
    # =============================================================
    add_body_p("The Class Diagram (Fig. 3) outlines the object-oriented structure of the software backend, including the SerialBridge, SessionEngine, HubertInferenceEngine, EventAggregator, DatabaseManager, and DashboardController classes.", indent=False)
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\02_class.png",
        "Fig. 3. Class Diagram.",
        width_inches=5.2
    )

    # =============================================================
    # PAGE 13: Fig. 4 Speech Session Streaming Sequence
    # =============================================================
    add_body_p("The Speech Session Streaming Sequence Diagram (Fig. 4) depicts the hardware-bridge communication lifecycle, including port initialization, PING/PONG handshaking, continuous VXF1 frame transmission, CRC verification, and terminal frame finalization.", indent=False)
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\03_audio_sequence.png",
        "Fig. 4. Speech Session Streaming Sequence.",
        width_inches=3.6
    )

    # =============================================================
    # PAGE 14: Fig. 5 Session Analysis Sequence
    # =============================================================
    add_body_p("The Session Analysis Sequence Diagram (Fig. 5) details the backend workflow upon receiving a session analyze request: audio validation, window slicing, HuBERT-D inference, event aggregation, database storage, and JSON response generation.", indent=False)
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\04_prediction_sequence.png",
        "Fig. 5. Session Analysis Sequence.",
        width_inches=5.2
    )

    # =============================================================
    # PAGE 15: Fig. 6 Activity Diagram
    # =============================================================
    add_body_p("The Activity Diagram (Fig. 6) models the unified operational workflow, illustrating how both live hardware capture and direct file upload pathways converge into standardized audio validation, inference, and visualization.", indent=False)
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\06_activity.png",
        "Fig. 6. Activity Diagram.",
        width_inches=1.8
    )

    # =============================================================
    # PAGE 16: Fig. 7 State Machine Diagram
    # =============================================================
    add_body_p("The State Machine Diagram (Fig. 7) defines the operational states of the capture unit and backend, transitioning from IDLE to WAITING_FOR_START, RECORDING, RECONSTRUCTING, ANALYZING, and COMPLETE.", indent=False)
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\07_state_machine.png",
        "Fig. 7. State Machine Diagram.",
        width_inches=3.2
    )

    # =============================================================
    # PAGE 17: 6.4 Deployment Design & Fig. 8 Deployment Diagram
    # =============================================================
    add_section_heading("6.4 Deployment Design", page_break_before=False)
    add_body_p("The Deployment Diagram (Fig. 8) details the physical execution nodes. The ESP32 capture unit acts as an edge recording peripheral connected via USB-UART to the host workstation. The host workstation runs the Python runtime hosting the Serial Bridge, Flask REST API (port 5000), Streamlit UI (port 8501), and local SQLite database. No external cloud infrastructure is required at runtime.")
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\09_deployment.png",
        "Fig. 8. Deployment Diagram.",
        width_inches=3.2
    )

    # =============================================================
    # PAGE 18: 6.5 Database Design & Fig. 9 ER Diagram
    # =============================================================
    add_section_heading("6.5 Database Design", page_break_before=False)
    add_body_p("The Entity-Relationship (ER) Diagram (Fig. 9) illustrates the relational database schema in app/voxflow.db. The schema consists of three normalized tables: (1) sessions, storing session-level metadata, duration, and summary event counts; (2) window_predictions, storing per-window timestamps and probability scores for repetition, prolongation, and block; and (3) aggregated_events, storing consolidated clinical events with start/end bounds, duration, and peak confidence.")
    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\uml\11_er_database.png",
        "Fig. 9. ER / Database Diagram.",
        width_inches=4.6
    )

    # =============================================================
    # PAGE 19: CHAPTER 7: METHODOLOGY (7.1, 7.2 & Fig. 10)
    # =============================================================
    add_chapter_heading("7. METHODOLOGY", page_break_before=False)
    
    add_section_heading("7.1 Identified Methodologies & End-to-End Workflow Overview")
    add_body_p("The VoxFlow methodology establishes a complete 12-step processing pipeline from physical acoustic capture to visual clinical analytics: (1) physical acoustic digitization, (2) I2S DMA buffering, (3) VXF1 framing, (4) serial transmission, (5) frame reception and CRC verification, (6) WAV file reconstruction, (7) audio validation, (8) sliding-window segmentation, (9) HuBERT-D neural inference, (10) temporal event aggregation, (11) relational database persistence, and (12) visual dashboard analytics.")

    add_section_heading("7.2 Hardware Capture and Framing")
    add_body_p("The Hardware Capture and Transmission Flow is illustrated in Fig. 10. The ESP32 samples audio at 16,000 Hz (16-bit mono PCM). Every 64 ms (1,024 samples = 2,048 bytes), the firmware constructs a VXF1 packet containing a 16-byte header: 4-byte Magic (0x56 0x58 0x46 0x31 / 'VXF1'), 4-byte Sequence Number (0, 1, 2...), 4-byte Payload Length (2,048 bytes), and 4-byte CRC32 checksum, as shown in Table 6. A terminal frame (Length = 0, Sequence = 0xFFFFFFFF) signals normal recording completion.")
    
    add_caption("Table 6. VXF1 Binary Protocol Frame Layout")
    add_styled_table(
        ["Field", "Offset", "Size", "Type", "Description"],
        [
            ["Magic", "0", "4 bytes", "uint8_t[4]", "ASCII magic identifier: 'V' 'X' 'F' '1' (0x56 0x58 0x46 0x31)"],
            ["Sequence No.", "4", "4 bytes", "uint32_t (LE)", "Monotonically increasing packet index (0, 1, 2...)"],
            ["Payload Length", "8", "4 bytes", "uint32_t (LE)", "Payload size in bytes (2,048 for audio; 0 for terminal frame)"],
            ["CRC32", "12", "4 bytes", "uint32_t (LE)", "IEEE 802.3 CRC32 checksum of the payload bytes"],
            ["Audio Payload", "16", "N bytes", "int16_t[]", "Raw 16 kHz 16-bit signed PCM mono audio samples"]
        ],
        col_widths=[1.2, 0.8, 0.9, 1.3, 1.8]
    )

    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\methodology\1_AudioCapture_Transmission.png",
        "Fig. 10. Hardware Capture and Transmission Flow.",
        width_inches=1.8
    )

    # =============================================================
    # PAGE 20: 7.3 Serial Bridge & 7.4 Audio Preprocessing
    # =============================================================
    add_section_heading("7.3 Serial Bridge and WAV Reconstruction", page_break_before=False)
    add_body_p("The Python serial bridge (hardware/bridge/serial_bridge.py) reads incoming bytes from the COM port at 460,800 baud. It searches for the 4-byte 'VXF1' magic header, unpacks the 16-byte header, verifies that the sequence number matches the expected increment, and computes the CRC32 checksum over the payload using zlib.crc32. Valid payloads are accumulated in a byte buffer. Upon receiving the terminal frame, the bridge packages the buffer into a standardized WAV file (16 kHz, 16-bit mono) using soundfile and writes it to recordings/.")

    add_section_heading("7.4 Audio Preprocessing and Windowing")
    add_body_p("Upon receiving a WAV file, the backend validates audio parameters (16 kHz sampling rate, mono channel, duration between 3.0 and 60.0 seconds, RMS silence threshold >= 0.0001). The audio waveform is normalized to [-1.0, 1.0] and sliced into 3.0-second sliding windows (48,000 samples) with a 1.0-second hop (16,000 samples). For an audio of duration D seconds, the number of windows generated is N = floor(D - 3.0) + 1. For example, a 10-second recording generates exactly 8 sliding windows.")

    # =============================================================
    # PAGE 21: 7.5 HuBERT-D Fluency Prediction & Fig. 11
    # =============================================================
    add_section_heading("7.5 HuBERT-D Fluency Prediction", page_break_before=False)
    add_body_p("The HuBERT-D Speech Fluency Prediction Flow is illustrated in Fig. 11. Each 3-second audio window is processed by HuBERT-D (facebook/hubert-base-ls960 backbone fine-tuned with a multi-label classification head). The model outputs a 3-dimensional logit vector passed through a sigmoid activation function, yielding independent probabilities: P_rep, P_pro, and P_blk. Predictions are thresholded using calibrated validation cutoffs: Repetition = 0.77, Prolongation = 0.79, and Block = 0.57.")

    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\methodology\2_SpeechFluency_Prediction.png",
        "Fig. 11. HuBERT-D Speech Fluency Prediction Flow.",
        width_inches=2.0
    )

    # =============================================================
    # PAGE 22: 7.6 Event Aggregation, 7.7 Summaries, 7.8 Dashboard & Fig. 12
    # =============================================================
    add_section_heading("7.6 Temporal Event Aggregation", page_break_before=False)
    add_body_p("Because sliding windows overlap by 2.0 seconds, a single disfluent event spanning across adjacent windows produces multiple positive window detections. VoxFlow applies a temporal event aggregation algorithm (app/services/event_aggregator.py): for each disfluency class, positive windows that overlap or lie within a merge gap threshold (gap <= 1.5 seconds) are merged into a single clinical event. The event start time is set to the first window's start, the end time to the last window's end, and the event confidence to the maximum probability observed across supporting windows.")

    add_section_heading("7.7 Session Summaries and Metrics")
    add_body_p("Following event aggregation, the backend computes session-level summary statistics: total recording duration, effective speech duration, total disfluency count, disfluency rate (events per minute of speech), and per-class event breakdowns. These summary metrics provide speech therapists with immediate, standardized clinical indices.")

    add_section_heading("7.8 Dashboard and Visual Analytics")
    add_body_p("The Prediction Storage and Dashboard Flow is illustrated in Fig. 12. Session metrics, window predictions, and aggregated events are persisted to SQLite. The Streamlit dashboard queries these records to render interactive Plotly timeline charts (showing exactly where repetitions, prolongations, and blocks occurred), per-class distribution bar charts, and longitudinal trend plots tracking disfluency rates across multiple sessions.")

    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\methodology\3_PredStorage_Dashboard.png",
        "Fig. 12. Prediction Storage and Dashboard Flow.",
        width_inches=3.0
    )

    # =============================================================
    # PAGE 23: 7.9 Algorithms 1, 2, 3
    # =============================================================
    add_section_heading("7.9 Algorithms", page_break_before=False)
    add_body_p("The core operational logic of VoxFlow is formally defined by three primary algorithms implemented in the repository:")

    add_algorithm_box(
        "1", "Audio Capture and VXF1 Streaming (Firmware)",
        [
            "1:  Initialize I2S peripheral (16 kHz, 16-bit, mono) and USB-UART (460,800 baud)",
            "2:  Create FreeRTOS ring queue Q of capacity 16 frames",
            "3:  Wait for START signal (GPIO27 button press or 'R' command over serial)",
            "4:  Set state = RECORDING, seq = 0, total_samples = 0, enable LED blinking (2 Hz)",
            "5:  while state == RECORDING do",
            "6:      Read 1,024 samples (2,048 bytes) from I2S DMA buffer; Push into queue Q",
            "7:      if Q is not empty then",
            "8:          Pop buffer; Compute crc = CRC32(audio_buffer)",
            "9:          Construct 16-byte header: magic='VXF1', seq=seq, len=2048, crc=crc",
            "10:         Transmit header + audio_buffer over USB-UART",
            "11:         seq = seq + 1, total_samples = total_samples + 1,024",
            "12:     end if",
            "13:     if STOP button pressed OR 'S' received OR total_samples >= 960,000 (60.0s) then state = STOPPED",
            "14: end while",
            "15: Transmit terminal frame: magic='VXF1', seq=0xFFFFFFFF, len=0, crc=0; Transmit 'DONE'"
        ]
    )

    add_algorithm_box(
        "2", "Session Speech Analysis (Backend)",
        [
            "1:  Input: session_audio_path (WAV file), session_id",
            "2:  Load audio waveform x, sampling_rate fs; Validate fs == 16000, 3.0s <= duration(x) <= 60.0s, RMS >= 0.0001",
            "3:  window_size = 48,000 samples (3.0s), hop_size = 16,000 samples (1.0s)",
            "4:  N = floor((length(x) - window_size) / hop_size) + 1; Initialize list window_predictions = []",
            "5:  for i = 0 to N - 1 do",
            "6:      start = i * hop_size, end = start + window_size; window_audio = x[start : end]",
            "7:      [p_rep, p_pro, p_blk] = HuBERT_D_Inference(window_audio)",
            "8:      window_predictions.append({t_start: start/fs, t_end: end/fs, p_rep, p_pro, p_blk})",
            "9:  end for",
            "10: events = Temporal_Event_Aggregation(window_predictions); summary = Compute_Session_Summary(events)",
            "11: Persist_To_Database(session_id, summary, window_predictions, events); return summary and events"
        ]
    )

    add_algorithm_box(
        "3", "Temporal Event Aggregation (Event Consolidation)",
        [
            "1:  Input: window_predictions, thresholds {T_rep=0.77, T_pro=0.79, T_blk=0.57}, gap=1.5s",
            "2:  Initialize list aggregated_events = []",
            "3:  for each disfluency class c in [Repetition, Prolongation, Block] do",
            "4:      Filter active windows W_c = [w in window_predictions if w.prob[c] >= thresholds[c]]",
            "5:      if W_c is empty then continue; Sort W_c chronologically by w.t_start",
            "6:      cluster = [W_c[0]]",
            "7:      for k = 1 to length(W_c) - 1 do",
            "8:          if W_c[k].t_start - cluster.last().t_end <= gap then cluster.append(W_c[k])",
            "9:          else aggregated_events.append(Create_Event(c, cluster)); cluster = [W_c[k]]",
            "10:     end for",
            "11:     aggregated_events.append(Create_Event(c, cluster))",
            "12: end for",
            "13: Sort aggregated_events chronologically by event.start_time; return aggregated_events"
        ]
    )

    # =============================================================
    # PAGE 24: CHAPTER 8: HARDWARE IMPLEMENTATION (8.1, 8.2, Fig. 13 & 8.3)
    # =============================================================
    add_chapter_heading("8. HARDWARE IMPLEMENTATION", page_break_before=False)
    
    add_section_heading("8.1 Hardware Development & Component Selection")
    add_body_p("The hardware capture subsystem is developed around the ESP32 Dev Module (ESP-WROOM-32) and the INMP441 MEMS digital microphone. The ESP32 was chosen for its integrated hardware I2S peripheral, dual-core architecture allowing decoupled audio acquisition and serial transmission, and high-speed USB-UART bridge. The INMP441 was selected because it provides digital pulse code modulation (PCM) output via I2S, offering a high Signal-to-Noise Ratio (61 dBA) and flat frequency response without the noise vulnerabilities of analogue electret capsules.")

    add_section_heading("8.2 Circuit and Wiring")
    add_body_p("The hardware schematic and physical pin connections between the ESP32 and peripheral components are detailed in Fig. 13 and Table 7. The INMP441 L/R pin is tied to GND to configure mono left-channel output. Two tactile push buttons with internal pull-ups provide START (GPIO27) and STOP (GPIO33) controls. A status LED connected to GPIO4 provides immediate operational visual feedback.")

    add_figure_image(
        r"c:\Users\user\Downloads\voxbox\v2\VoxFlow_uml\VoxFlow_uml\png\circuit_design\circuit_design.png",
        "Fig. 13. ESP32–INMP441 Circuit Diagram.",
        width_inches=3.6
    )

    add_caption("Table 7. ESP32 Pin Assignment and Peripheral Connections")
    add_styled_table(
        ["Peripheral", "Module Pin", "ESP32 GPIO", "Direction / Configuration", "Functional Role"],
        [
            ["INMP441", "VDD", "3V3 (3.3V)", "Power Output", "Regulated 3.3V DC power supply"],
            ["INMP441", "GND", "GND", "Power Ground", "Common electrical ground reference"],
            ["INMP441", "L/R", "GND", "Configuration", "Tied to GND for Left channel mono output"],
            ["INMP441", "SCK (BCLK)", "GPIO26", "ESP32 Output (Master)", "I2S Continuous Bit Clock (512 kHz)"],
            ["INMP441", "WS (LRCLK)", "GPIO25", "ESP32 Output (Master)", "I2S Word Select / Frame Sync (16 kHz)"],
            ["INMP441", "SD (DATA)", "GPIO32", "ESP32 Input", "I2S Serial Audio Data Stream"],
            ["Push Button 1", "START", "GPIO27", "Input (Pull-up)", "Active LOW trigger to start 3–60s recording"],
            ["Push Button 2", "STOP", "GPIO33", "Input (Pull-up)", "Active LOW trigger to terminate recording"],
            ["Status LED", "Anode (+)", "GPIO4", "Output (Current Limit)", "Solid = Standby/Ready, 2 Hz Blink = Recording"]
        ],
        col_widths=[1.1, 0.9, 1.0, 1.4, 1.6]
    )

    add_section_heading("8.3 Firmware Implementation")
    add_body_p("The firmware (hardware/firmware/voxflow_esp32_firmware/voxflow_esp32_firmware.ino) is implemented in C++ using the ESP32 Arduino Core and FreeRTOS. Key implementation highlights include:")
    add_bullet_p("FreeRTOS Task Pinning:", "The audio capture task (I2SCaptureTask) is pinned to Core 0 with an 8,192-byte stack running at maximum priority (configMAX_PRIORITIES - 1), ensuring zero audio sample dropping during CPU-intensive operations.")
    add_bullet_p("Queue-Based Audio Buffering:", "A FreeRTOS ring queue of 16 slots buffers 1,024-sample audio blocks between the producer task and the transmission loop on Core 1.")
    add_bullet_p("Framed VXF1 Packaging:", "The transmission loop constructs the 16-byte header, calculates the IEEE 802.3 CRC32 checksum, and transmits framed packets over USB-UART at 460,800 baud.")
    add_bullet_p("Safety Stop Mechanisms:", "Recording automatically terminates upon: (1) physical STOP button press (GPIO33), (2) receipt of ASCII 'S' command from host serial, or (3) reaching exactly 960,000 samples (60.00 seconds).")

    # =============================================================
    # PAGE 25: CHAPTER 9: SOFTWARE IMPLEMENTATION (9.1 to 9.7)
    # =============================================================
    add_chapter_heading("9. SOFTWARE IMPLEMENTATION", page_break_before=False)
    
    add_section_heading("9.1 Software Development Overview")
    add_body_p("The VoxFlow software ecosystem is organized into a modular architecture comprising the Serial Bridge, Session Processing Engine, HuBERT-D Inference Engine, Temporal Event Aggregator, Flask REST Backend, and Streamlit Dashboard.")

    add_section_heading("9.2 Serial Bridge Implementation")
    add_body_p("The serial bridge (hardware/bridge/serial_bridge.py) manages bidirectional communication with the ESP32 over serial. It verifies connectivity via PING/PONG handshakes, monitors the serial byte stream for the 'VXF1' magic bytes, parses packet headers, verifies sequential sequence counters, computes CRC32 checksums, and writes valid PCM streams into standardized 16 kHz mono WAV files.")

    add_section_heading("9.3 Session Processing Engine")
    add_body_p("The session engine (app/services/session_engine.py) encapsulates the complete session processing lifecycle. It implements validate_and_load_session_audio() to enforce audio standards, process_session_windows() to generate overlapping 3-second windows, and process_continuous_session() to orchestrate inference, aggregation, database storage, and response formatting.")

    add_section_heading("9.4 HuBERT-D Inference Engine")
    add_body_p("The inference engine (inference/inference_pipeline.py) loads the fine-tuned HuBERT-D model checkpoint. Audio windows are padded or trimmed to exactly 48,000 samples, normalized, and converted to PyTorch tensors. Forward inference produces logits that are mapped via sigmoid into per-class probabilities for repetition, prolongation, and block.")

    add_section_heading("9.5 Temporal Event Aggregator")
    add_body_p("The event aggregator (app/services/event_aggregator.py) implements the temporal clustering algorithm described in Algorithm 3. It groups contiguous positive window activations within a 1.5-second merge threshold into single, consolidated clinical events, eliminating duplicate counts across overlapping analysis windows.")

    add_section_heading("9.6 Flask REST API")
    add_body_p("The REST backend (server.py, app/api/session_routes.py) provides structured HTTP endpoints for client applications:")
    add_bullet_p("GET /api/v1/health:", "Returns system health, active device, and database status.")
    add_bullet_p("GET /api/v1/model/info:", "Returns loaded model architecture, checkpoint path, and calibrated detection thresholds.")
    add_bullet_p("POST /api/v1/session/start & stop:", "Controls hardware recording lifecycle via the serial bridge.")
    add_bullet_p("POST /api/v1/session/analyze:", "Accepts a WAV audio file or session ID, performs end-to-end analysis, and returns complete session metrics and events.")
    add_bullet_p("GET /api/v1/sessions & /api/v1/session/<id>:", "Retrieves stored session records, event breakdowns, and historical data.")

    add_section_heading("9.7 Streamlit Dashboard")
    add_body_p("The frontend dashboard (dashboard.py, app/ui/streamlit_app.py) provides a modern clinical visual analytics interface with four functional pages: (1) Live Hardware Capture, (2) Audio File Analysis, (3) Session Inspection & Timeline, and (4) Longitudinal History & Trends.")

    # =============================================================
    # PAGE 26: 9.8 CS INTEGRATION & CHAPTER 10: TESTING (10.1)
    # =============================================================
    add_section_heading("9.8 Data Analytics and Computer Science Integration", page_break_before=False)
    add_body_p("VoxFlow represents an advanced integration of modern computer science disciplines: digital signal processing (I2S DMA sampling, audio normalization, sliding-window framing), self-supervised deep learning (Transformer-based acoustic representation learning and multi-label inference), algorithmic event processing (temporal clustering and merge-gap aggregation), relational database engineering (normalized SQLite schema with referential integrity), RESTful web architecture (decoupled HTTP microservices), and visual analytics (interactive time-series charting and longitudinal trend tracking). The current prototype does not use a cloud service at runtime. It runs locally on the PC, ensuring patient privacy, low latency, and zero cloud hosting costs.")

    add_chapter_heading("10. TESTING AND VALIDATION")
    
    add_section_heading("10.1 Software Testing")
    add_body_p("VoxFlow incorporates a comprehensive automated test suite implemented in Python's standard unittest framework. The test suite comprises 38 automated test cases organized across five specialized test modules, covering the serial bridge, audio validation, sliding-window framing, event aggregation, database operations, and REST endpoints. All 38 automated tests pass successfully (38/38 passed), as summarized in Table 8.")

    add_caption("Table 8. Automated Software Test Suite Summary")
    add_styled_table(
        ["Test Module", "Focus Area", "Test Cases", "Status"],
        [
            ["tests/test_serial_bridge.py", "VXF1 framing, CRC32 verification, sequence validation", "8", "PASSED (8/8)"],
            ["tests/test_audio_validation.py", "Sampling rate, duration bounds (3–60s), silence detection", "7", "PASSED (7/7)"],
            ["tests/test_session_engine.py", "Sliding window generation (3s / 1s hop), hop math", "6", "PASSED (6/6)"],
            ["tests/test_event_aggregator.py", "Temporal clustering, 1.5s gap merge, multi-window consolidation", "7", "PASSED (7/7)"],
            ["tests/test_api_endpoints.py", "Flask REST endpoints, HTTP status codes, error handling", "10", "PASSED (10/10)"],
            ["Total Test Suite", "Full end-to-end automated software verification", "38", "PASSED (38/38)"]
        ],
        col_widths=[2.0, 2.5, 0.8, 0.9]
    )

    # =============================================================
    # PAGE 27: 10.2 to 10.6 (Physical Validation, Demo, Deployment)
    # =============================================================
    add_section_heading("10.2 API and Bridge Validation", page_break_before=False)
    add_body_p("API validation verified that malformed audio inputs are correctly identified and rejected: a 2.0-second WAV file (below the 3.0 s minimum) is rejected with HTTP 422 Unprocessable Entity, silent audio (RMS < 0.0001) is rejected with HTTP 422, and valid 5.0-second and 60.0-second WAV files successfully return HTTP 200 with structured JSON analysis results. Bridge validation confirmed that simulated corrupt VXF1 frames (corrupted CRC32 or skipped sequence numbers) are rejected immediately without writing corrupt audio to disk.")

    add_section_heading("10.3 Model Evaluation")
    add_body_p("HuBERT-D was evaluated on the held-out test split of 6,812 clips across 249 unseen speakers. Predictions were assessed using Macro F1, Mean ROC-AUC, and per-class F1 scores, confirming superior generalization over baseline models.")

    add_section_heading("10.4 Physical Hardware Validation")
    add_body_p("To maintain strict scientific honesty, software test results and physical hardware validation are explicitly separated. Physical hardware validation confirms electrical and acoustic functionality on the assembled prototype (Table 9).")

    add_caption("Table 9. Physical Hardware Validation Status")
    add_styled_table(
        ["Validation Check", "Procedure & What Was Checked", "Hardware Status"],
        [
            ["I2S Bit Clock & WS", "Verified 512 kHz SCK and 16 kHz WS frame sync on GPIO26/GPIO25", "VERIFIED"],
            ["Microphone Audio Sampling", "Recorded spoken speech with INMP441; verified clean 16 kHz PCM waveform", "VERIFIED"],
            ["VXF1 Serial Streaming", "Streamed audio at 460,800 baud over USB-UART with zero dropped frames", "VERIFIED"],
            ["START / STOP Buttons", "Tested physical push buttons on GPIO27 and GPIO33 for session control", "VERIFIED"],
            ["Status LED Indication", "Verified solid standby light and 2 Hz blinking during active recording", "VERIFIED"],
            ["60-Second Auto-Stop", "Verified firmware auto-terminates at 960,000 samples and sends terminal frame", "VERIFIED"]
        ],
        col_widths=[1.8, 3.4, 1.0]
    )

    add_section_heading("10.5 Functional Demonstration")
    add_body_p("The real operational workflow was validated across a complete 10-step demonstration sequence:")
    add_bullet_p("Step 1 (System Startup):", "Launch the Flask backend (python server.py) and Streamlit dashboard (streamlit run dashboard.py).")
    add_bullet_p("Step 2 (Hardware Connection):", "Connect the ESP32 capture unit via USB; the bridge detects COM port and performs PING/PONG handshake.")
    add_bullet_p("Step 3 (Initiate Recording):", "Press the physical START button (GPIO27) or click 'Start Recording' on the dashboard.")
    add_bullet_p("Step 4 (Speech Capture):", "Speak into the INMP441 microphone; the LED blinks at 2 Hz as audio streams in VXF1 frames.")
    add_bullet_p("Step 5 (Stop Recording):", "Press the STOP button (GPIO33) or allow the 60-second limit to expire.")
    add_bullet_p("Step 6 (WAV Reconstruction):", "The serial bridge verifies CRC32 checksums, detects the terminal frame, and writes the WAV file.")
    add_bullet_p("Step 7 (Automated Analysis):", "The bridge posts the WAV to /api/v1/session/analyze, triggering windowing and HuBERT-D inference.")
    add_bullet_p("Step 8 (Event Aggregation):", "Overlapping positive windows are merged into non-redundant clinical disfluency events.")
    add_bullet_p("Step 9 (Database Persistence):", "Session metadata, window probabilities, and aggregated events are stored in SQLite.")
    add_bullet_p("Step 10 (Clinical Review):", "The dashboard automatically renders the session timeline, event metrics, and longitudinal trends.")

    add_section_heading("10.6 Validation and Deployment")
    add_body_p("The deployment model operates entirely on-premise on the host workstation connected to the ESP32 capture unit via USB-UART (as shown in the Deployment Diagram, Fig. 8). The software stack requires no active internet connection or cloud runtime dependencies, ensuring total patient privacy, HIPAA compliance compatibility, zero cloud service costs, and deterministic sub-second processing latency.")

    # =============================================================
    # PAGE 28: CHAPTER 11: RESULTS AND DISCUSSION
    # =============================================================
    add_chapter_heading("11. RESULTS AND DISCUSSION", page_break_before=False)
    
    add_section_heading("11.1 Model Results")
    add_body_p("We evaluated three distinct model architectures on the speaker-exclusive test split (6,812 clips across 249 unseen speakers): (1) a traditional baseline utilizing 13 MFCCs with Support Vector Machines (MFCC + SVM), (2) a fine-tuned wav2vec 2.0 transformer, and (3) our fine-tuned HuBERT-D model. As presented in Table 10, HuBERT-D achieved the highest overall performance with a Macro F1 score of 0.4599 and a Mean ROC-AUC of 0.8099, outperforming wav2vec 2.0 (0.4498 / 0.8020) and substantially surpassing the MFCC + SVM baseline (0.3121 / 0.6633).")

    add_caption("Table 10. Model Evaluation Comparison on Speaker-Exclusive Test Split")
    add_styled_table(
        ["Model Architecture", "Input Features", "Macro F1", "Mean ROC-AUC", "Repetition F1", "Prolongation F1", "Block F1"],
        [
            ["MFCC + SVM (Baseline)", "13 MFCCs + delta + delta-delta", "0.3121", "0.6633", "0.4102", "0.3215", "0.2046"],
            ["wav2vec 2.0 (Fine-tuned)", "Raw Audio (16 kHz)", "0.4498", "0.8020", "0.5412", "0.4730", "0.3352"],
            ["HuBERT-D (Proposed)", "Raw Audio (16 kHz)", "0.4599", "0.8099", "0.5565", "0.4841", "0.3393"]
        ],
        col_widths=[1.7, 1.3, 0.7, 0.8, 0.7, 0.7, 0.7]
    )

    add_body_p("A granular analysis of HuBERT-D per-class performance reveals that Repetition detection achieved the highest F1 score (0.5565), followed by Prolongation (0.4841), while Block detection remained the most challenging class (0.3393). This pattern aligns with clinical reality: repetitions and prolongations exhibit salient acoustic and phonetic cues (rhythmic repetition and sustained formants), whereas blocks are characterized by silent postural fixations that are acoustically difficult to distinguish from natural conversational pauses.")

    add_section_heading("11.2 Session Analysis Results")
    add_body_p("Session processing tests confirm that sliding-window analysis paired with temporal event aggregation correctly measures continuous speech sessions. Slicing a 10-second speech recording produces 8 windows; when simulated repetitions occur across windows 2, 3, and 4 (timestamps 1.0s to 6.0s), the temporal aggregator consolidates the 3 positive windows into exactly 1 unified clinical event spanning from 1.0s to 6.0s with peak confidence. This verifies that event consolidation prevents artificial inflation of clinical disfluency counts.")

    add_section_heading("11.3 System Discussion")
    add_body_p("The experimental and operational results confirm that VoxFlow achieves its primary design goals: reliable digital capture via ESP32, error-checked transport via VXF1, robust self-supervised disfluency detection via HuBERT-D, and clean visual analytics via Streamlit. Operating entirely on local PC hardware without cloud dependencies guarantees data privacy and eliminates operational costs.")

    # =============================================================
    # PAGE 29: CHAPTER 12: LIMITATIONS & FUTURE SCOPE & CHAPTER 13: CONCLUSION
    # =============================================================
    add_chapter_heading("12. LIMITATIONS AND FUTURE SCOPE", page_break_before=False)
    
    add_section_heading("12.1 Limitations")
    add_body_p("While VoxFlow demonstrates a successful end-to-end implementation, several limitations are acknowledged:")
    add_bullet_p("Block Detection Sensitivity:", "Block detection achieved an F1 of 0.3393. Silent blocks remain difficult to separate from natural pauses without visual articulatory tracking or lexical language model context.")
    add_bullet_p("Acoustic Domain Mismatch:", "HuBERT-D was fine-tuned on podcast and studio recordings (SEP-28k / FluencyBank). Although the INMP441 provides high-quality digital audio, acoustic domain adaptation may improve performance under background noise.")
    add_bullet_p("Single-User Local Deployment:", "The prototype is designed for a single workstation without multi-user authentication or role-based clinical access control.")

    add_section_heading("12.2 Future Scope")
    add_body_p("Future development will focus on the following enhancements:")
    add_bullet_p("Multimodal Block Detection:", "Integrate facial landmark and articulatory motion tracking via webcam to detect silent oral fixations, significantly boosting block detection accuracy.")
    add_bullet_p("ASR & Lexical Integration:", "Incorporate automatic speech recognition (Whisper) to align acoustic disfluency predictions with text transcripts for syllable-accurate %SS calculation.")
    add_bullet_p("Clinical Pilot Trials:", "Conduct formal clinical validation studies in collaboration with speech-language pathologists to assess usability and clinical utility in real therapeutic practice.")
    add_bullet_p("Edge AI Inference:", "Explore quantized on-device inference using ESP32-S3 or low-power edge accelerators for fully standalone pocket analyzers.")

    add_chapter_heading("13. CONCLUSION")
    add_body_p("In this project, we designed, implemented, and verified VoxFlow, an end-to-end speech fluency and disfluency session analyzer. VoxFlow bridges the gap between theoretical machine learning clip classification and practical clinical session assessment by uniting low-cost embedded hardware capture (ESP32 + INMP441), an error-checked serial protocol (VXF1 with CRC32 verification), self-supervised deep learning modeling (HuBERT-D), algorithmic temporal event aggregation, relational SQLite persistence, and an interactive Streamlit visual analytics dashboard.")
    add_body_p("Evaluated under strict speaker-exclusive constraints across 519 distinct speakers, HuBERT-D achieved a Macro F1 score of 0.4599 and a Mean ROC-AUC of 0.8099 on unseen voices, outperforming wav2vec 2.0 (0.4498) and an MFCC + SVM baseline (0.3121). The temporal event aggregator successfully consolidates overlapping sliding-window detections into bounded clinical events, eliminating redundant multi-counting. The entire software suite passes all 38 automated unit and integration tests. Operating entirely on local PC hardware without cloud runtime dependencies, VoxFlow provides an accessible, privacy-preserving, and scientifically grounded tool for automated speech fluency analysis.")

    # =============================================================
    # PAGE 30: REFERENCES & APPENDIX A
    # =============================================================
    add_chapter_heading("REFERENCES", page_break_before=False)
    
    references = [
        "[1] S. A. Sheikh, M. Sahidullah, F. Hirsch and S. Ouni, \"Machine learning for stuttering identification: Review, challenges and future directions,\" Neurocomputing, vol. 514, pp. 385–402, 2022, doi: 10.1016/j.neucom.2022.10.015.",
        "[2] C. Lea, V. Mitra, A. Joshi, S. Kajarekar and J. P. Bigham, \"SEP-28k: A dataset for stuttering event detection from podcasts with people who stutter,\" in Proc. IEEE ICASSP, 2021, pp. 6798–6802, doi: 10.1109/ICASSP39728.2021.9413520.",
        "[3] N. Bernstein Ratner and B. MacWhinney, \"Fluency Bank: A new resource for fluency research and practice,\" Journal of Fluency Disorders, vol. 56, pp. 69–80, 2018, doi: 10.1016/j.jfludis.2018.03.002.",
        "[4] P. Howell, S. Davis and J. Bartrip, \"The University College London Archive of Stuttered Speech (UCLASS),\" Journal of Speech, Language, and Hearing Research, vol. 52, no. 2, pp. 556–569, 2009, doi: 10.1044/1092-4388(2009/07-0129).",
        "[5] T. Kourkounakis, A. Hajavi and A. Etemad, \"FluentNet: End-to-end detection of stuttered speech disfluencies with deep learning,\" IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 2986–2999, 2021, doi: 10.1109/TASLP.2021.3110146.",
        "[6] S. A. Sheikh, M. Sahidullah, F. Hirsch and S. Ouni, \"StutterNet: Stuttering detection using time delay neural network,\" in Proc. 29th European Signal Processing Conference (EUSIPCO), 2021, pp. 426–430, doi: 10.23919/EUSIPCO54536.2021.9616063.",
        "[7] S. A. Sheikh, M. Sahidullah, F. Hirsch and S. Ouni, \"Advancing stuttering detection via data augmentation, class-balanced loss and multi-contextual deep learning,\" IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 5, pp. 2553–2564, 2023, doi: 10.1109/JBHI.2023.3248281.",
        "[8] S. P. Bayerl, A. Wolff von Gudenberg, F. Hönig, E. Nöth and K. Riedhammer, \"KSoF: The Kassel State of Fluency dataset – A therapy centered dataset of stuttering,\" in Proc. 13th Language Resources and Evaluation Conference (LREC), 2022, pp. 1780–1787. [Online]. Available: https://aclanthology.org/2022.lrec-1.189/",
        "[9] S. P. Bayerl, D. Wagner, E. Nöth and K. Riedhammer, \"Detecting dysfluencies in stuttering therapy using wav2vec 2.0,\" in Proc. Interspeech, 2022, pp. 2868–2872, doi: 10.21437/Interspeech.2022-10908.",
        "[10] S. P. Bayerl, D. Wagner, E. Nöth, T. Bocklet and K. Riedhammer, \"The influence of dataset partitioning on dysfluency detection systems,\" in Text, Speech, and Dialogue (TSD 2022), Lecture Notes in Computer Science, Springer, 2022, pp. 423–436, doi: 10.1007/978-3-031-16270-1_35.",
        "[11] S. P. Bayerl, M. Gerczuk, A. Batliner, C. Bergler, S. Amiriparian, B. Schuller, E. Nöth and K. Riedhammer, \"Classification of stuttering – The ComParE challenge and beyond,\" Computer Speech & Language, vol. 81, art. 101519, 2023, doi: 10.1016/j.csl.2023.101519.",
        "[12] A. Baevski, Y. Zhou, A. Mohamed and M. Auli, \"wav2vec 2.0: A framework for self-supervised learning of speech representations,\" in Advances in Neural Information Processing Systems 33 (NeurIPS), 2020, pp. 12449–12460.",
        "[13] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov and A. Mohamed, \"HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,\" IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021, doi: 10.1109/TASLP.2021.3122291.",
        "[14] VoxFlow Project Team, \"VoxFlow — Speech Fluency & Disfluency Session Analyzer,\" GitHub Repository, Oct. 2026. [Online]. Available: https://github.com/PetaSivaNandhanReddy/voxflow",
        "[15] InvenSense (TDK), \"INMP441: Omnidirectional Microphone with Bottom Port and I2S Digital Output,\" Product Data Sheet, Rev. 1.1.",
        "[16] Espressif Systems, \"ESP32 Series Datasheet & Technical Reference Manual,\" Espressif Systems, 2024. [Online]. Available: https://www.espressif.com/",
        "[17] Meta AI, \"facebook/hubert-base-ls960 Model Card,\" Hugging Face, 2021. [Online]. Available: https://huggingface.co/facebook/hubert-base-ls960"
    ]

    for ref in references:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        run = p.add_run(ref)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(8.5)

    add_chapter_heading("APPENDICES")
    
    add_section_heading("Appendix A — Key Configuration Parameters")
    add_caption("Table 11. Key System Configuration Parameters")
    add_styled_table(
        ["Configuration Setting", "Configured Value", "Defined Location / Module"],
        [
            ["Sampling Rate / Audio Format", "16,000 Hz, 16-bit signed PCM, Mono", "Firmware (I2S), configs/config.py"],
            ["I2S Frame Size / FreeRTOS Queue", "1,024 samples (2,048 bytes) / 16 slots", "Firmware (voxflow_esp32_firmware.ino)"],
            ["Session Duration Limits", "Min 3.0 s, Max 60.0 s (960,000 samples)", "Firmware, Serial Bridge, Session Engine"],
            ["Serial Baud Rate & Protocol", "460,800 baud, VXF1 framed binary", "Firmware, hardware/bridge/serial_bridge.py"],
            ["Sliding Window / Hop Size", "3.0 s (48,000 samples) / 1.0 s (16,000 samples)", "configs/config.py, Session Engine"],
            ["Decision Thresholds", "Repetition 0.77, Prolongation 0.79, Block 0.57", "configs/config.py, models/Hubert_D/val_summary.json"],
            ["Temporal Event Merge Gap", "1.5 seconds (maximum cluster window gap)", "app/services/event_aggregator.py"],
            ["RMS Silence Threshold", "RMS < 0.0001 (triggers HTTP 422)", "app/services/session_engine.py"],
            ["Database File Path", "app/voxflow.db (SQLite relational schema)", "configs/config.py, DatabaseManager"],
            ["Microservice Network Ports", "Flask Backend: 5000, Streamlit UI: 8501", "configs/config.py, Streamlit default"]
        ],
        col_widths=[2.0, 2.2, 2.0]
    )

    # =============================================================
    # PAGE 31: APPENDIX B & APPENDIX C
    # =============================================================
    add_section_heading("Appendix B — Testing and Validation Evidence Status", page_break_before=False)
    add_body_p("All 38 automated unit and integration tests execute and pass (38/38 passed). The complete test execution logs, API response payloads (confirming HTTP 422 rejection on 2-second and silent audio, and HTTP 200 success on 5-second and 60-second audio), and UI screenshots are documented in the companion Testing & Validation Evidence document.")
    add_body_p("Hardware validation assets (oscilloscope clock verification, physical push button triggering, status LED blinking, and 60-second automatic limit enforcement) are verified on the physical prototype.")

    add_section_heading("Appendix C — Supporting Documents and Repository Assets")
    add_body_p("The complete VoxFlow engineering artifacts—including ESP32 C++ firmware, Python serial bridge, Flask backend, HuBERT-D inference engine, Streamlit dashboard, automated test suites, and PlantUML diagram source models—are maintained in the official repository: https://github.com/PetaSivaNandhanReddy/voxflow.")

    return doc

if __name__ == '__main__':
    doc = create_report()
    
    # Save to reports&ppt
    docx_path1 = os.path.abspath(r"c:\Users\user\Downloads\voxbox\reports&ppt\VoxFlow_Final_Project_Report.docx")
    doc.save(docx_path1)
    print("Saved DOCX to:", docx_path1)
    
    # Also save to v2/docs
    docs_dir = os.path.abspath(r"c:\Users\user\Downloads\voxbox\v2\docs")
    os.makedirs(docs_dir, exist_ok=True)
    docx_path2 = os.path.join(docs_dir, "VoxFlow_Final_Project_Report.docx")
    doc.save(docx_path2)
    print("Saved DOCX to:", docx_path2)
