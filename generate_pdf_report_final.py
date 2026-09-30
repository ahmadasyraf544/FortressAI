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

PDF_EN_PATH = r"d:\SEDIC_2026\SEDIC_2026_Final_Training_Report.pdf"
PDF_MY_PATH = r"d:\SEDIC_2026\LAPORAN_AKHIR_PRESTASI_SEDIC_2026.pdf"

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
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header (Skip page 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "SEDIC 2026 - Final AI Model Training & Accuracy Benchmark Report")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY - SEDIC 2026 MARITIME AI PROJECT")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        
        self.restoreState()

def create_report(file_path):
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Color Palette
    NAVY = colors.HexColor("#1A365D")
    BLUE = colors.HexColor("#2B6CB0")
    GREEN = colors.HexColor("#276749")
    GOLD = colors.HexColor("#D69E2E")
    TEXT_DARK = colors.HexColor("#2D3748")
    BG_LIGHT = colors.HexColor("#F7FAFC")
    BG_GREEN = colors.HexColor("#F0FFF4")
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        textColor=NAVY, alignment=0, spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=15,
        textColor=BLUE, spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=14, leading=18,
        textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=14,
        textColor=BLUE, spaceBefore=9, spaceAfter=5, keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark', parent=styles['BodyText'],
        fontName='Helvetica', fontSize=9, leading=13.5,
        textColor=TEXT_DARK, spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletDark', parent=body_style,
        leftIndent=12, firstLineIndent=-8, spaceAfter=3
    )

    badge_style = ParagraphStyle(
        'BadgeText', parent=body_style,
        fontName='Helvetica-Bold', fontSize=10, leading=14,
        textColor=GREEN, alignment=1
    )

    code_style = ParagraphStyle(
        'CodeSnippet', parent=body_style,
        fontName='Courier', fontSize=8, leading=10,
        textColor=colors.HexColor("#C53030"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5, borderPadding=5,
        spaceBefore=3, spaceAfter=6
    )

    story = []

    # =========================================================================
    # HEADER & TITLE BANNER
    # =========================================================================
    story.append(Paragraph("SEDIC 2026: Maritime Vessel Detection System", title_style))
    story.append(Paragraph("Final AI Model Training Benchmark & Comprehensive Technical Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceBefore=0, spaceAfter=12))

    # Highlight Achievement Box
    achieve_box = [
        [
            Paragraph("<b>TARGET ACCURACY (>80.00%) EXCEEDED!</b><br/>"
                      "<b>Achieved mAP@50: 91.36%</b> &nbsp;|&nbsp; "
                      "<b>Precision: 87.75%</b> &nbsp;|&nbsp; "
                      "<b>Recall: 85.09%</b> &nbsp;|&nbsp; "
                      "<b>mAP@50-95: 79.98%</b>", badge_style)
        ]
    ]
    t_achieve = Table(achieve_box, colWidths=[504])
    t_achieve.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_GREEN),
        ('BOX', (0, 0), (-1, -1), 1.5, GREEN),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_achieve)
    story.append(Spacer(1, 10))

    # Meta Table
    meta_data = [
        [Paragraph("<b>Project:</b> SEDIC 2026 Vessel Detection", body_style), Paragraph("<b>Completion Date:</b> September 13, 2026", body_style)],
        [Paragraph("<b>Model Architecture:</b> YOLOv8 Nano (yolov8n.pt)", body_style), Paragraph("<b>Hardware:</b> NVIDIA GTX 1650 Ti (4GB VRAM)", body_style)],
        [Paragraph("<b>Training Duration:</b> 50 Epochs Complete", body_style), Paragraph("<b>Total Dataset:</b> 49,128 Annotated Images", body_style)],
        [Paragraph("<b>Inference Speed:</b> ~16.0 ms/frame (~60 FPS)", body_style), Paragraph("<b>Saved Model Weights:</b> runs/detect/.../best.pt", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))

    # =========================================================================
    # 1. EXECUTIVE SUMMARY & ACCURACY BENCHMARK
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Accuracy Benchmark", h1_style))
    story.append(Paragraph(
        "The 50-epoch deep learning training pipeline for the <b>SEDIC 2026 Maritime Vessel Detection System</b> "
        "has completed successfully. The project target was to achieve an object detection accuracy threshold of <b>80.00% and above</b>. "
        "Upon completing 50 full epochs on the 49,128-image dataset (39,303 training / 9,825 validation), the model achieved an "
        "outstanding <b>mAP@50 score of 91.36%</b>, surpassing the target requirement by <b>+11.36%</b>.",
        body_style
    ))
    story.append(Paragraph(
        "Additionally, the model demonstrated exceptional multi-class classification and spatial localization accuracy across all 10 target maritime vessel classes, "
        "achieving a <b>Precision of 87.75%</b> and a <b>Recall of 85.09%</b>.",
        body_style
    ))

    # Benchmark Progress Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Table 1.1: Empirical Accuracy Metric Progression across 50 Training Epochs</b>", h2_style))
    
    prog_data = [
        ["Epoch", "Train Box Loss", "Train Class Loss", "Precision", "Recall", "mAP@50", "mAP@50-95", "Performance Note"],
        ["1", "0.9829", "1.8699", "74.83%", "68.50%", "73.45%", "51.24%", "Initial warm-up & feature initialization"],
        ["10", "0.7240", "0.7484", "83.98%", "82.51%", "87.86%", "72.07%", "Passed 80% mAP@50 threshold"],
        ["25", "0.5981", "0.6123", "86.45%", "84.14%", "90.09%", "77.48%", "Crossed 90% mAP@50 milestone"],
        ["40", "0.5277", "0.5386", "86.89%", "84.75%", "90.74%", "78.72%", "Stable convergence & fine-tuning"],
        ["50 (Final)", "0.4371", "0.4396", "87.75%", "85.09%", "91.36%", "79.98%", "PEAK MODEL ACCURACY (best.pt)"]
    ]
    t_prog = Table(prog_data, colWidths=[40, 65, 75, 55, 55, 55, 60, 99])
    t_prog.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7.5),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, BG_LIGHT]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#C6F6D5")),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_prog)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 2. END-TO-END SYSTEM PROCESS FLOW
    # =========================================================================
    story.append(Paragraph("2. End-to-End System Process Flow", h1_style))
    story.append(Paragraph(
        "The complete operational pipeline of the SEDIC 2026 system consists of 6 sequential processing phases:",
        body_style
    ))

    flow_steps = [
        ["Phase", "Phase Name", "Primary Script / Tool", "Core Processing Function & Output"],
        ["1", "Data Acquisition & Auto-Labeling", "autolabel.py / autolabel_images.py", "Automated zero-shot image labeling using YOLO-World model to generate initial .txt annotations."],
        ["2", "Quality Assurance & Data Cleaning", "LabelImg & preview.py", "Human-in-the-loop verification to correct mislabeled or missing bounding boxes (QA/QC)."],
        ["3", "Dataset Partitioning & Config", "prepare_train_val_split()", "Automated 80% Train (39,303) / 20% Val (9,825) directory split & data.yaml registration."],
        ["4", "AI Model Training & Optimization", "train.py (YOLOv8n)", "CNN feature extraction, loss computation, backpropagation & GTX 1650 Ti VRAM optimization."],
        ["5", "Model Evaluation & Weights Export", "Ultralytics Engine", "Serializing best.pt weights, generating Confusion Matrix & Precision-Recall curves."],
        ["6", "Video Inference & Deployment", "predict.py", "Real-time frame-by-frame vessel detection on video surveillance feeds & MP4 export."]
    ]
    t_flow = Table(flow_steps, colWidths=[35, 130, 130, 209])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_flow)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. DETAILED TECHNICAL & HARDWARE OPTIMIZATION
    # =========================================================================
    story.append(Paragraph("3. Hardware Optimization & Training Stability Rationale", h1_style))
    story.append(Paragraph(
        "Training deep neural networks on high-resolution datasets requires strict hardware resource management. "
        "To ensure execution stability on the target hardware (<b>NVIDIA GeForce GTX 1650 Ti with 4GB VRAM</b>) on Windows, "
        "the following parameters were engineered in <code>train.py</code>:",
        body_style
    ))

    train_params = [
        ["Parameter", "Configured Value", "Technical Rationale & Engineering Impact"],
        ["Base Architecture", "yolov8n.pt", "Nano architecture (3.0M params, 8.2 GFLOPs) provides fast inference (~16ms/frame) ideal for video stream processing."],
        ["Batch Size", "16", "Reduced from 32 to 16 to fit comfortably within the 4GB VRAM limit without triggering CUDA Out-of-Memory (OOM)."],
        ["Image Resolution", "512px", "Optimal spatial resolution maintaining vessel structural clarity while capping GPU VRAM allocation."],
        ["RAM Caching", "cache=False", "Disabled RAM caching because caching 49k+ images requires >24GB RAM. Direct disk reading prevents system MemoryError."],
        ["DataLoader Workers", "workers=2", "Reduced from 4 to 2 workers to prevent Windows IPC pipe memory crashes during PyTorch multiprocessing spawn."],
        ["AMP Precision", "amp=False", "Disabled FP16 Automatic Mixed Precision because GTX 1650 Ti lacks dedicated Tensor Cores, preventing NaN loss."]
    ]
    t_train = Table(train_params, colWidths=[85, 75, 344])
    t_train.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_train)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Loss Convergence Analysis:</b>", h2_style))
    story.append(Paragraph("• <b>Training Box Loss:</b> Dropped from <b>0.9829</b> to <b>0.4371</b> (-55.5% reduction), demonstrating precise bounding box alignment.", bullet_style))
    story.append(Paragraph("• <b>Training Class Loss:</b> Dropped from <b>1.8699</b> to <b>0.4396</b> (-76.5% reduction), showing sharp classification separation between vessel categories.", bullet_style))
    story.append(Paragraph("• <b>Validation Loss:</b> Validation Box Loss reached <b>0.4461</b> and Validation Class Loss reached <b>0.4268</b> with no overfitting divergence.", bullet_style))

    story.append(Spacer(1, 10))

    # =========================================================================
    # 4. TARGET VESSEL CLASS SPECIFICATIONS
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
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_classes)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 5. VIDEO INFERENCE & CONCLUSION
    # =========================================================================
    story.append(Paragraph("5. Real-Time Video Inference & Final Conclusion", h1_style))
    story.append(Paragraph(
        "Upon completing training, the <code>predict.py</code> script loaded the optimal weights (<code>best.pt</code>) "
        "and processed 18 maritime surveillance video files in <code>videos/</code>. The model achieved real-time detection "
        "speeds of <b>~16.0 ms per frame (~60 FPS)</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Conclusion:</b> The SEDIC 2026 Maritime Vessel Detection System has successfully surpassed all performance "
        "requirements, achieving <b>91.36% mAP@50 accuracy</b>. All trained weights, metric plots, and annotated video outputs "
        "are saved and ready for deployment.",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Report Generated Successfully At: {file_path}")

if __name__ == "__main__":
    create_report(PDF_EN_PATH)
    create_report(PDF_MY_PATH)
