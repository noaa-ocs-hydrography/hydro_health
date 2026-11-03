

import concurrent.futures
import logging
import os
import random
import s3fs
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import rasterio

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

matplotlib.use("Agg")
GDAL_S3_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}


class DigitalCoastValidationReport:
    """
    Audits S3 GeoTIFF outputs, builds an Executive Summary, renders decimated samples,
    and outputs a comprehensive verification PDF.
    """

    def __init__(
        self,
        s3_prefix: str,
        execution_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        fs: Optional[s3fs.S3FileSystem] = None,
    ):
        """Initialize the validator with storage filesystem and report styles."""

        self.fs = fs or s3fs.S3FileSystem(anon=False)
        self.raw_prefix = s3_prefix
        self.clean_prefix = s3_prefix.replace("s3://", "").strip("/")
        self.execution_metrics = execution_metrics or {}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.styles = getSampleStyleSheet()

    def _get_single_crs(self, file_path: str) -> str:
        """Read only the CRS header of a single raster using byte-range queries."""

        try:
            with rasterio.Env(**GDAL_S3_ENV):
                with self.fs.open(file_path, "rb") as s3_file:
                    with rasterio.open(s3_file) as src:
                        if src.crs:
                            epsg = src.crs.to_epsg()
                            return f"EPSG:{epsg}" if epsg else str(src.crs)
                        return "Missing CRS"
        except Exception as e:
            return f"Read Error ({type(e).__name__})"

    def _render_sample_image(self, provider: str, chosen_file: str) -> Optional[str]:
        """Stream a decimated low-resolution raster thumbnail directly over the wire."""

        try:
            with rasterio.Env(**GDAL_S3_ENV):
                with self.fs.open(chosen_file, "rb") as s3_file:
                    with rasterio.open(s3_file) as src:
                        max_dim = 400
                        scale = min(max_dim / src.width, max_dim / src.height, 1.0)
                        out_shape = (
                            1,
                            max(1, int(src.height * scale)),
                            max(1, int(src.width * scale)),
                        )
                        data = src.read(1, out_shape=out_shape, masked=True)

                        fig, ax = plt.subplots(figsize=(5.5, 2.8), dpi=120)
                        im = ax.imshow(data, cmap="viridis")
                        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)
                        ax.set_title(
                            f"Sample Decimated Read: {chosen_file.split('/')[-1]}",
                            fontsize=8,
                            pad=6,
                        )
                        ax.tick_params(labelsize=7)

                        preview_path = os.path.join(self.temp_dir.name, f"{provider}_preview.png")
                        fig.tight_layout()
                        fig.savefig(preview_path, bbox_inches="tight")
                        plt.close(fig)
                        return preview_path
        except Exception as e:
            logging.warning(f"Failed to generate thumbnail for {chosen_file}: {e}")
            return None

    def _inspect_provider_rasters(
        self, provider: str, files: List[str], max_workers: int = 16
    ) -> Tuple[List[str], Optional[str], Optional[str]]:
        """Extract all unique CRSs concurrently and build a single visual preview."""

        if not files:
            return ["None detected"], None, "N/A"

        chosen_file = random.choice(files)
        preview_image_path = self._render_sample_image(provider, chosen_file)

        crs_set: Set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for crs_result in executor.map(self._get_single_crs, files):
                crs_set.add(crs_result)

        return sorted(list(crs_set)), preview_image_path, chosen_file.split("/")[-1]

    def _collect_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Walk S3 prefix to gather actual uploaded files, counts, and byte sizes."""

        all_objects = self.fs.find(self.clean_prefix, detail=True)
        providers = defaultdict(lambda: {"count": 0, "size_bytes": 0, "files": []})

        for path, meta in all_objects.items():
            if path.lower().endswith((".tif", ".tiff")):
                rel_path = path[len(self.clean_prefix):].strip("/")
                parts = rel_path.split("/")
                provider = parts[0] if len(parts) > 1 else "root"

                providers[provider]["count"] += 1
                providers[provider]["size_bytes"] += meta.get("size", 0)
                providers[provider]["files"].append(path)

        return dict(providers)

    def generate_pdf(
        self, output_pdf_filename: str = "DigitalCoast_Validation_Report.pdf"
    ) -> str:
        """Compile executive audit table, provider summaries, and thumbnails into a PDF."""

        inventory = self._collect_inventory()
        all_providers = sorted(list(set(inventory.keys()) | set(self.execution_metrics.keys())))
        if not all_providers:
            raise FileNotFoundError(
                f"No GeoTIFFs or logged runs found for prefix s3://{self.clean_prefix}"
            )

        doc = SimpleDocTemplate(
            output_pdf_filename,
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )

        story = []

        title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Title"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1A365D"),
            alignment=0,
        )
        meta_style = ParagraphStyle(
            "MetaText", parent=self.styles["Normal"], fontSize=9, leading=13
        )
        h2_style = ParagraphStyle(
            "H2",
            parent=self.styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1A365D"),
        )
        provider_title_style = ParagraphStyle(
            "ProviderTitle",
            parent=self.styles["Heading3"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2B6CB0"),
        )
        tbl_head_style = ParagraphStyle(
            "TblHead",
            parent=self.styles["Normal"],
            fontSize=8.5,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
        tbl_cell_style = ParagraphStyle(
            "TblCell", parent=self.styles["Normal"], fontSize=8, leading=10
        )
        tbl_pass_style = ParagraphStyle(
            "TblPass",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#22543D"),
            fontName="Helvetica-Bold",
        )
        tbl_fail_style = ParagraphStyle(
            "TblFail",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#9B2C2C"),
            fontName="Helvetica-Bold",
        )

        run_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Header Section
        story.append(Paragraph("DigitalCoast Ingestion & Validation Audit", title_style))
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            Paragraph(
                f"<b>S3 Target Prefix:</b> s3://{self.clean_prefix}<br/>"
                f"<b>Timestamp:</b> {run_date} | <b>Monitored Providers:</b> {len(all_providers)}",
                meta_style,
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Executive Summary Table
        story.append(Paragraph("Execution Summary Audit", h2_style))
        story.append(Spacer(1, 0.05 * inch))

        summary_table_data = [
            [
                Paragraph("Provider", tbl_head_style),
                Paragraph("Target (Exp)", tbl_head_style),
                Paragraph("Actual (S3)", tbl_head_style),
                Paragraph("Volume", tbl_head_style),
                Paragraph("Status / Verdict", tbl_head_style),
            ]
        ]

        total_expected = 0
        total_downloaded = 0
        total_bytes = 0

        for provider in all_providers:
            info = inventory.get(provider, {"count": 0, "size_bytes": 0, "files": []})
            exec_info = self.execution_metrics.get(provider, {})

            actual_cnt = info["count"]
            expected_cnt = exec_info.get("expected", actual_cnt)
            size_mb = info["size_bytes"] / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb / 1024:.2f} GB"

            total_expected += expected_cnt
            total_downloaded += actual_cnt
            total_bytes += info["size_bytes"]

            if expected_cnt == actual_cnt and actual_cnt > 0 and exec_info.get("failed", 0) == 0:
                verdict = Paragraph("OK (100%)", tbl_pass_style)
            elif actual_cnt == 0:
                verdict = Paragraph("FAILED (0 files)", tbl_fail_style)
            else:
                missing = expected_cnt - actual_cnt
                verdict = Paragraph(
                    f"MISMATCH (-{missing} files)" if missing > 0 else "PARTIAL ANOMALY",
                    tbl_fail_style,
                )

            summary_table_data.append(
                [
                    Paragraph(f"<b>{provider}</b>", tbl_cell_style),
                    Paragraph(str(expected_cnt), tbl_cell_style),
                    Paragraph(str(actual_cnt), tbl_cell_style),
                    Paragraph(size_str, tbl_cell_style),
                    verdict,
                ]
            )

        total_mb = total_bytes / (1024 * 1024)
        total_size_label = f"{total_mb:.1f} MB" if total_mb < 1024 else f"{total_mb / 1024:.2f} GB"
        overall_status = (
            "ALL PASS"
            if total_expected == total_downloaded and total_downloaded > 0
            else "AUDIT DEFICIT"
        )

        summary_table_data.append(
            [
                Paragraph("<b>TOTALS</b>", tbl_cell_style),
                Paragraph(f"<b>{total_expected}</b>", tbl_cell_style),
                Paragraph(f"<b>{total_downloaded}</b>", tbl_cell_style),
                Paragraph(f"<b>{total_size_label}</b>", tbl_cell_style),
                Paragraph(
                    f"<b>{overall_status}</b>",
                    tbl_pass_style if overall_status == "ALL PASS" else tbl_fail_style,
                ),
            ]
        )

        summary_table = Table(
            summary_table_data,
            colWidths=[2.6 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch, 1.8 * inch],
        )
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EDF2F7")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F7FAFC")]),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 0.25 * inch))

        # Detailed Provider Inspection
        story.append(Paragraph("Detailed Provider Inspection", h2_style))
        story.append(Spacer(1, 0.05 * inch))

        for provider in all_providers:
            info = inventory.get(provider, {"count": 0, "size_bytes": 0, "files": []})
            exec_info = self.execution_metrics.get(provider, {})

            crs_list, img_path, sample_file = self._inspect_provider_rasters(
                provider, info["files"]
            )
            crs_display = ", ".join(crs_list) if crs_list else "None detected"
            size_mb = info["size_bytes"] / (1024 * 1024)
            size_label = (
                f"{size_mb:.2f} MB" if size_mb < 1024 else f"{size_mb / 1024:.2f} GB"
            )

            provider_rows = [
                [
                    Paragraph("<b>Target Expected:</b>", tbl_cell_style),
                    Paragraph(str(exec_info.get("expected", "N/A")), tbl_cell_style),
                ],
                [
                    Paragraph("<b>Downloaded (S3):</b>", tbl_cell_style),
                    Paragraph(f"{info['count']} rasters ({size_label})", tbl_cell_style),
                ],
                [
                    Paragraph("<b>CRS Detected:</b>", tbl_cell_style),
                    Paragraph(crs_display, tbl_cell_style),
                ],
                [
                    Paragraph("<b>Sample Raster:</b>", tbl_cell_style),
                    Paragraph(sample_file, tbl_cell_style),
                ],
            ]

            if exec_info.get("errors"):
                err_preview = "; ".join(exec_info["errors"][:3])
                if len(exec_info["errors"]) > 3:
                    err_preview += f" ... (+{len(exec_info['errors']) - 3} more)"
                provider_rows.append(
                    [
                        Paragraph("<b>Logged Errors:</b>", tbl_cell_style),
                        Paragraph(err_preview, tbl_fail_style),
                    ]
                )

            card_table = Table(provider_rows, colWidths=[1.5 * inch, 6.0 * inch])
            card_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )

            elements = [
                Paragraph(f"Provider: <b>{provider}</b>", provider_title_style),
                Spacer(1, 0.04 * inch),
                card_table,
                Spacer(1, 0.08 * inch),
            ]

            if img_path and os.path.exists(img_path):
                elements.append(RLImage(img_path, width=4.8 * inch, height=2.4 * inch))

            elements.append(Spacer(1, 0.2 * inch))
            story.append(KeepTogether(elements))

        doc.build(story)
        return output_pdf_filename

    def upload_to_s3(
        self, local_pdf_path: str, destination_s3_uri: Optional[str] = None
    ) -> str:
        """Upload the generated PDF to S3."""

        filename = os.path.basename(local_pdf_path)
        target_path = (
            destination_s3_uri.replace("s3://", "").strip("/")
            if destination_s3_uri
            else f"{self.clean_prefix}/reports/{filename}"
        )
        self.fs.put(local_pdf_path, target_path)
        return f"s3://{target_path}"

    def cleanup(self) -> None:
        """Release temporary preview images from disk."""

        self.temp_dir.cleanup()