import os
import pandas as pd
from typing import List, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class PDFReportGenerator:
    """
    Executive PDF Report Generator built with ReportLab.
    Creates a styled 2-page document containing executive insights, 
    key visual charts, and a complete feature selection audit table.
    """

    def __init__(self, output_pdf_path: str = "Key_Drivers_Executive_Report.pdf"):
        self.pdf_path = output_pdf_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Defines modern typography and color schemes."""
        self.styles.add(ParagraphStyle(
            'DocTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1A365D'),
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=17,
            textColor=colors.HexColor('#2B6CB0'),
            spaceBefore=14,
            spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            'ExecutiveBody',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor('#2D3748'),
            spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            'TableHeader',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=0
        ))
        self.styles.add(ParagraphStyle(
            'TableCell',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor('#2D3748')
        ))

    def generate_report(
        self,
        target_col: str,
        target_type: str,
        initial_feature_count: int,
        isolated_drivers: List[str],
        audit_df: pd.DataFrame,
        chart_paths: Dict[str, str]
    ) -> str:
        """
        Assembles and writes the PDF file to disk.
        """
        doc = SimpleDocTemplate(
            self.pdf_path,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        elements = []

        # -----------------------------------------------------------------
        # PAGE 1: TITLE, EXECUTIVE SUMMARY & VISUALIZATIONS
        # -----------------------------------------------------------------
        elements.append(Paragraph("Executive Key Drivers & Selection Audit Report", self.styles['DocTitle']))
        
        meta_info = (
            f"<b>Target Attribute Analyzed:</b> {target_col} ({target_type.capitalize()}) | "
            f"<b>Total Features Evaluated:</b> {initial_feature_count}"
        )
        elements.append(Paragraph(meta_info, self.styles['ExecutiveBody']))
        elements.append(Spacer(1, 6))

        # Executive Summary Box
        top_driver_str = f"<b>'{isolated_drivers[0]}'</b>" if isolated_drivers else "N/A"
        summary_text = (
            f"<b>Key Findings:</b> Out of <b>{initial_feature_count}</b> initial metrics evaluated, "
            f"the pipeline isolated <b>{len(isolated_drivers)} primary driver(s)</b>: "
            f"<code>{', '.join(isolated_drivers)}</code>. "
            f"Primary driver {top_driver_str} demonstrates the strongest overall relationship with the target."
        )
        summary_table = Table([[Paragraph(summary_text, self.styles['ExecutiveBody'])]], colWidths=[540])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#BEE3F8')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 10))

        # Chart Section
        elements.append(Paragraph("1. Primary Business Driver Visualisations", self.styles['SectionHeading']))
        
        # Grid layout for Importance Bar Chart & Correlation Heatmap
        img_table_data = [
            [
                Image(chart_paths["importance"], width=3.6*inch, height=1.5*inch),
                Image(chart_paths["heatmap"], width=3.6*inch, height=1.5*inch)
            ]
        ]
        img_table = Table(img_table_data, colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 8))

        # Direct Trend / Distribution Chart
        elements.append(Image(chart_paths["trend"], width=7.2*inch, height=2.2*inch))
        elements.append(Spacer(1, 10))

        # -----------------------------------------------------------------
        # PAGE 2: FEATURE SELECTION AUDIT LOG
        # -----------------------------------------------------------------
        elements.append(PageBreak())

        elements.append(Paragraph("2. Complete Feature Selection Audit Log", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "The table below documents the exact mathematical stage, metric value, and justification "
            "for every variable evaluated during feature selection:", 
            self.styles['ExecutiveBody']
        ))
        elements.append(Spacer(1, 6))

        # Build Audit Log Table
        table_data = [[
            Paragraph("Feature Name", self.styles['TableHeader']),
            Paragraph("Status", self.styles['TableHeader']),
            Paragraph("Selection Stage", self.styles['TableHeader']),
            Paragraph("Metric Value", self.styles['TableHeader']),
            Paragraph("Elimination Justification", self.styles['TableHeader'])
        ]]

        for _, row in audit_df.iterrows():
            # Format status styling
            is_retained = row['status'].lower() == 'retained'
            status_color = "#2B6CB0" if is_retained else "#C53030"
            status_html = f"<font color='{status_color}'><b>{row['status']}</b></font>"

            table_data.append([
                Paragraph(str(row["feature"]), self.styles['TableCell']),
                Paragraph(status_html, self.styles['TableCell']),
                Paragraph(str(row["stage"]), self.styles['TableCell']),
                Paragraph(str(row["metric_value"]), self.styles['TableCell']),
                Paragraph(str(row["reason"]), self.styles['TableCell'])
            ])

        # Column width allocation adding up to 540pt (7.5 inches printable width)
        audit_table = Table(table_data, colWidths=[100, 60, 110, 80, 190], repeatRows=1)
        audit_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))

        elements.append(audit_table)
        
        # Build Document
        doc.build(elements)
        print(f"PDF successfully generated: {os.path.abspath(self.pdf_path)}")
        return self.pdf_path