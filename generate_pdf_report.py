import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = r"d:\SEDIC_2026\SEDIC_2026_Process_Flow_Report.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (Skip page 1 cover)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "SEDIC 2026 - AI System Process & Technical Workflow Specification")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
            
        # Footer (All pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY - SEDIC 2026 PROJECT")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1A365D")   # Deep Navy
    SECONDARY = colors.HexColor("#2B6CB0") # Steel Blue
    ACCENT = colors.HexColor("#D69E2E")    # Warm Gold
    TEXT_DARK = colors.HexColor("#2D3748") # Dark Slate Body Text
    BG_LIGHT = colors.HexColor("#F7FAFC")  # Soft Grey/White
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        alignment=0,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=TEXT_DARK,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=body_style,
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#C53030"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # DOCUMENT HEADER
    # =========================================================================
    story.append(Paragraph("SEDIC 2026: Maritime Vessel Detection System", title_style))
    story.append(Paragraph("Technical Specification Report & End-to-End AI Process Flow", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=0, spaceAfter=15))

    # Metadata Table
    meta_data = [
        [Paragraph("<b>Project:</b> SEDIC 2026 Vessel Detection", body_style), Paragraph("<b>Date:</b> September 13, 2026", body_style)],
        [Paragraph("<b>AI Architecture:</b> YOLOv8 Nano (yolov8n.pt)", body_style), Paragraph("<b>GPU Hardware:</b> NVIDIA GTX 1650 Ti (4GB VRAM)", body_style)],
        [Paragraph("<b>Target Classes:</b> 10 Maritime Vessel Classes", body_style), Paragraph("<b>Training Status:</b> Stable Optimised Pipeline", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This document provides a comprehensive technical breakdown of the system architecture and process flow "
        "for the <b>SEDIC 2026 Maritime Vessel Detection & Recognition System</b>. The system leverages state-of-the-art "
        "Computer Vision and Deep Learning techniques powered by <b>YOLOv8 (You Only Look Once v8)</b> to achieve "
        "real-time detection, tracking, and multi-class classification of maritime vessels across surveillance video feeds and imagery.",
        body_style
    ))
    story.append(Paragraph(
        "The end-to-end operational pipeline encompasses 6 core interconnected phases: automated zero-shot data labeling, "
        "quality assurance and data cleaning, dataset partitioning, hardware-optimized AI model training, quantitative "
        "performance evaluation, and real-time video inference deployment.",
        body_style
    ))

    # =========================================================================
    # 2. END-TO-END PROCESS FLOW
    # =========================================================================
    story.append(Paragraph("2. End-to-End System Process Flow", h1_style))
    
    flow_steps = [
        ["Phase", "Phase Name", "Primary Script / Tool", "Core Processing Function & Output"],
        ["1", "Data Acquisition & Auto-Labeling", "autolabel.py / autolabel_images.py", "Image acquisition and automated zero-shot labeling via YOLO-World."],
        ["2", "Quality Assurance & Data Cleaning", "LabelImg & preview.py", "Manual inspection and bounding box verification to ensure dataset integrity."],
        ["3", "Dataset Partitioning & Config", "prepare_train_val_split()", "Automated 80% Train : 20% Validation split and data.yaml configuration generation."],
        ["4", "AI Model Training & Optimization", "train.py (YOLOv8n)", "Neural network training, feature extraction, loss calculation, and backpropagation."],
        ["5", "Model Evaluation & Weights Export", "Ultralytics Engine", "Export of best.pt weights, Confusion Matrix analysis, and mAP accuracy plotting."],
        ["6", "Video Inference & Deployment", "predict.py", "Real-time vessel detection on input video feeds and annotated video output export."]
    ]
    t_flow = Table(flow_steps, colWidths=[35, 130, 130, 209])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_flow)
    story.append(Spacer(1, 15))

    # =========================================================================
    # 3. PHASE-BY-PHASE TECHNICAL BREAKDOWN
    # =========================================================================
    story.append(Paragraph("3. Detailed Phase-by-Phase Technical Breakdown", h1_style))

    # Phase 1
    story.append(Paragraph("Phase 1: Data Acquisition & Automated Labeling (Auto-Labeling)", h2_style))
    story.append(Paragraph(
        "Manually annotating tens of thousands of maritime images is time-prohibitive. Thus, <code>autolabel.py</code> and "
        "<code>autolabel_images.py</code> utilize <b>YOLO-World (Zero-Shot Object Detection)</b> to automatically annotate raw images.",
        body_style
    ))
    story.append(Paragraph("• <b>Processing Pipeline:</b> Raw imagery is fed into the pretrained zero-shot detector conditioned on 10 vessel class prompt descriptions.", bullet_style))
    story.append(Paragraph("• <b>Generated Output:</b> Standardized YOLO text annotation files (<code>.txt</code>) containing normalized coordinates:", bullet_style))
    story.append(Paragraph("&lt;class_id&gt; &lt;x_center&gt; &lt;y_center&gt; &lt;width&gt; &lt;height&gt;", code_style))

    # Phase 2
    story.append(Paragraph("Phase 2: Quality Assurance & Data Cleaning (Data QA/QC)", h2_style))
    story.append(Paragraph(
        "AI model performance is bound by data quality (<i>Garbage In, Garbage Out</i>). A verification phase was implemented to refine auto-generated labels.",
        body_style
    ))
    story.append(Paragraph("• <b>LabelImg GUI:</b> Allows human annotators to review, adjust, or delete misaligned or missing bounding boxes.", bullet_style))
    story.append(Paragraph("• <b>preview.py:</b> Lightweight CLI visualizer that rapidly overlays color-coded bounding boxes on images without launching heavy GUI applications.", bullet_style))

    # Phase 3
    story.append(Paragraph("Phase 3: Dataset Partitioning & Configuration (Dataset Split & Config)", h2_style))
    story.append(Paragraph(
        "The <code>prepare_train_val_split()</code> routine inside <code>train.py</code> automatically organizes flat image and label directories into standard YOLO format:",
        body_style
    ))
    story.append(Paragraph("• <b>80% Training Set (39,303 images):</b> Used by the neural network to optimize weights and learn visual features.", bullet_style))
    story.append(Paragraph("• <b>20% Validation Set (9,825 images):</b> Held-out dataset used to evaluate model generalization and prevent overfitting.", bullet_style))
    story.append(Paragraph("• <b>data.yaml:</b> Master dataset descriptor file mapping directory paths and the 10 target maritime class names.", bullet_style))

    story.append(Spacer(1, 10))

    # Phase 4 - Training & Hardware Optimization
    story.append(Paragraph("Phase 4: AI Model Training & Hardware Optimization", h2_style))
    story.append(Paragraph(
        "This phase forms the core computational engine where the <b>YOLOv8 Nano (yolov8n.pt)</b> model is trained. "
        "Specific parameter adjustments were engineered to ensure peak stability on an <b>NVIDIA GeForce GTX 1650 Ti (4GB VRAM)</b> GPU on Windows:",
        body_style
    ))

    train_params = [
        ["Parameter", "Value", "Technical Rationale & Impact"],
        ["Base Architecture", "yolov8n.pt", "Ultra-lightweight architecture (3.0M parameters, 8.2 GFLOPs) optimized for high-FPS real-time video detection."],
        ["Batch Size", "16", "Adjusted from 32 to 16 to prevent CUDA Out-of-Memory (OOM) errors on 4GB VRAM."],
        ["Image Resolution", "512px", "Optimal spatial resolution balancing feature resolution against VRAM allocation."],
        ["RAM Caching", "cache=False", "Disabled because caching 49k+ images requires >24GB RAM. Direct disk reading prevents Windows MemoryError."],
        ["DataLoader Workers", "workers=2", "Reduced from 4 to 2 to eliminate IPC pipe crashes during PyTorch multiprocessing spawn on Windows."],
        ["AMP Precision", "amp=False", "Disabled because GTX 1650 Ti lacks dedicated FP16 Tensor Cores, preventing NaN loss and check failures."]
    ]
    t_train = Table(train_params, colWidths=[85, 70, 349])
    t_train.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_train)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Mathematical Engine & Training Loop Execution:</b>", body_style))
    story.append(Paragraph("1. <b>Forward Pass:</b> Input batches pass through Convolutional Neural Network (CNN) layers, extracting multiscale spatial features via C2f and SPPF modules.", bullet_style))
    story.append(Paragraph("2. <b>Loss Computation:</b> Computes bounding box regression loss (<i>CIoU Loss</i>) and classification loss (<i>Binary Cross-Entropy Loss</i>).", bullet_style))
    story.append(Paragraph("3. <b>Backpropagation & Optimizer:</b> The MuSGD optimizer updates network weights to minimize total loss across epochs.", bullet_style))
    story.append(Paragraph("4. <b>Early Stopping (Patience=15):</b> Halts training automatically if validation mAP fails to improve for 15 consecutive epochs, preventing overfitting.", bullet_style))

    # Phase 5
    story.append(Paragraph("Phase 5: Performance Evaluation & Model Export", h2_style))
    story.append(Paragraph(
        "Training artifacts are automatically logged under <code>runs/detect/sedic_vessel_detector_50ep_b16_n/</code>. Key output files include:",
        body_style
    ))
    story.append(Paragraph("• <b>weights/best.pt:</b> Serialized model weights achieving the highest mean Average Precision (mAP) score.", bullet_style))
    story.append(Paragraph("• <b>results.csv & results.png:</b> Quantitative logs and trend plots for Loss, Precision, Recall, and mAP@0.5 / mAP@0.5:0.95.", bullet_style))
    story.append(Paragraph("• <b>confusion_matrix.png:</b> Inter-class confusion matrix identifying classification ambiguities between vessel types.", bullet_style))

    # Phase 6
    story.append(Paragraph("Phase 6: Real-Time Video Inference & Deployment (predict.py)", h2_style))
    story.append(Paragraph(
        "The <code>predict.py</code> script loads <code>best.pt</code> to execute end-to-end video inference:",
        body_style
    ))
    story.append(Paragraph("1. Frame Decoding: Input surveillance videos from <code>videos/</code> are decoded frame-by-frame.", bullet_style))
    story.append(Paragraph("2. Real-Time Detection: Model predicts bounding box coordinates, vessel class predictions, and confidence percentage scores (%).", bullet_style))
    story.append(Paragraph("3. Visual Rendering: Overlays color-coded bounding boxes and text callouts onto the video frames.", bullet_style))
    story.append(Paragraph("4. Video Export: Encodes and exports the final annotated video file (<code>.mp4</code>) for presentation and reporting.", bullet_style))

    story.append(Spacer(1, 15))

    # =========================================================================
    # 4. TARGET VESSEL CLASS SPECIFICATION TABLE
    # =========================================================================
    story.append(Paragraph("4. Target Maritime Vessel Class Specifications (10 Classes)", h1_style))

    classes_data = [
        ["ID", "Class Name", "Category", "Visual Identification Features & Characteristics"],
        ["0", "container_ship", "Commercial", "Large cargo vessels carrying stacked intermodal shipping containers."],
        ["1", "tanker", "Commercial", "Oil/chemical tankers featuring flat decks, piping manifolds, or storage tanks."],
        ["2", "cargo_ship", "Commercial", "Bulk carriers, freighters, or general cargo transport vessels."],
        ["3", "passenger_ferry", "Passenger", "Water taxis, passenger ferries, cruise liners, or commuter passenger vessels."],
        ["4", "yacht", "Recreational", "Luxury motor yachts, private sailing yachts, or pleasure craft."],
        ["5", "speedboat", "High-Speed", "Small high-speed motorboats, Rigid Inflatable Boats (RHIBs), or jet boats."],
        ["6", "fishing_boat", "Fishery", "Commercial trawlers, traditional fishing boats, sampans, or longline vessels."],
        ["7", "military_vessel_local", "Naval", "Domestic naval or Coast Guard vessels (grey hull, pennant numbers)."],
        ["8", "military_vessel_foreign", "Naval", "Foreign warships, destroyers, frigates, corvettes, or aircraft carriers."],
        ["9", "military_vessel_general", "Naval", "Unidentified, general, or auxiliary naval military vessels."]
    ]
    t_classes = Table(classes_data, colWidths=[25, 125, 80, 274])
    t_classes.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_classes)
    story.append(Spacer(1, 15))

    # =========================================================================
    # 5. CONCLUSION & TECHNICAL SUMMARY
    # =========================================================================
    story.append(Paragraph("5. Technical Conclusion & Summary", h1_style))
    story.append(Paragraph(
        "The SEDIC 2026 Maritime Vessel Detection System delivers a robust, scalable, and fully automated computer vision pipeline. "
        "By adopting YOLOv8 Nano, the system achieves an ideal balance between real-time inference speed and high detection accuracy.",
        body_style
    ))
    story.append(Paragraph(
        "The hardware optimizations engineered in <code>train.py</code> (batch size 16, disabled RAM caching, controlled worker threads) "
        "allow full pipeline execution on mid-range workstation hardware (NVIDIA GTX 1650 Ti 4GB VRAM) without memory fragmentation or crash faults.",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] English PDF Report Generated Successfully At: {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
