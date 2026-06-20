"""
ingestion/parser.py
PyMuPDF + YOLO + Table Transformer based document parser for RAG pipeline.
Outputs ParseResult with DocumentMeta + BlockRecord list for chunker.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
import tempfile
import time
import asyncio
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import fitz
import numpy as np
import cv2
import pandas as pd
import torch
import warnings

from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from doclayout_yolo import YOLOv10

from final_rag.config import (
    SUMMARY_MODEL,
    SUMMARY_OLLAMA_URL,
    SUMMARY_MAX_CHARS,
    YOLO_MODEL_PATH,
    TRANSFORMER_MODEL_PATH,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion.parser")


# ── Constants ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS     = {".pdf", ".docx", ".pptx", ".md"}
MAX_FILE_SIZE_MB         = 50.0
PARSE_TIMEOUT_SEC        = 300

TABLE_CROP_PADDING           = 20
TABLE_TRANSFORMER_THRESHOLD  = 0.5
TABLE_DEDUP_IOU_THRESHOLD    = 0.85
CELL_SHRINK_MARGIN_PTS       = max(1.5, 3.0 * 72.0 / 150.0)
MAX_NEAREST_CELL_DIST_SQ     = 2500.0
COMPLEX_TEXT_THRESHOLD       = 60
COMPLEX_TOTAL_THRESHOLD      = 80

_BOX_XMIN, _BOX_YMIN, _BOX_XMAX, _BOX_YMAX = 0, 1, 2, 3

_FIGURE_GARBAGE_RE = re.compile(
    r"(\[FIGURE\]|<!-- image -->)\s*\n((?:.{0,120}\n){1,10})",
    re.MULTILINE,
)


# ── Enums ──────────────────────────────────────────────────────────────
class ExtractionMethod(str, Enum):
    YOLO    = "yolo"
    SKIPPED = "skipped"
    ERROR   = "error"


# ── DocumentMeta ───────────────────────────────────────────────────────
@dataclass
class DocumentMeta:
    doc_id:          str
    file_name:       str
    file_path:       str
    file_type:       str
    file_size_kb:    float
    page_count:      int       = 0
    has_tables:      bool      = False
    parse_success:   bool      = True
    filename_tokens: list[str] = field(default_factory=list)
    doc_year:        str       = ""
    summary:         str       = ""
    keywords:        list[str] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)


# ── PageRecord ─────────────────────────────────────────────────────────
@dataclass
class PageRecord:
    page_no:    int
    page_label: str
    text:       str
    char_start: int = 0
    char_end:   int = 0


# ── BlockRecord ────────────────────────────────────────────────────────
@dataclass
class BlockRecord:
    block_type: str
    content:    str
    page_no:    int
    page_label: str  = ""
    section:    str  = ""
    is_table:   bool = False


# ── ParseResult ────────────────────────────────────────────────────────
@dataclass
class ParseResult:
    file_name:           str
    file_type:           str
    method_used:         ExtractionMethod
    markdown:            str
    meta:                Optional[DocumentMeta] = None
    total_pages:         int                    = 0
    processing_time_sec: float                  = 0.0
    error:               Optional[str]          = None
    success:             bool                   = True
    warnings:            list[str]              = field(default_factory=list)
    pages:               list[PageRecord]       = field(default_factory=list)
    blocks:              list[BlockRecord]      = field(default_factory=list)
    doc_id:              str                    = ""
    doc_year:            str                    = ""
    filename_tokens:     list[str]              = field(default_factory=list)
    summary:             str                    = ""
    keywords:            list[str]              = field(default_factory=list)


# ── Model loading ──────────────────────────────────────────────────────
def _get_device() -> str:
    if torch.cuda.is_available():
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
        return "cuda:0"
    logger.info("No GPU — using CPU")
    return "cpu"

DEVICE = _get_device()

logger.info("Loading YOLOv10 layout model...")
yolo_model = YOLOv10(str(YOLO_MODEL_PATH))
logger.info("Loading Table Transformer model...")
table_processor = AutoImageProcessor.from_pretrained(str(TRANSFORMER_MODEL_PATH))
table_model     = TableTransformerForObjectDetection.from_pretrained(str(TRANSFORMER_MODEL_PATH))


# ── Layout helpers ─────────────────────────────────────────────────────
def _get_dynamic_reading_order(boxes_data: list) -> list:
    if not boxes_data:
        return []
    boxes_data.sort(key=lambda b: b[2])
    rows = []
    for box in boxes_data:
        idx, x1, y1, x2, y2 = box
        placed = False
        for row in rows:
            row_y1   = min(b[2] for b in row)
            row_y2   = max(b[4] for b in row)
            overlap  = max(0, min(y2, row_y2) - max(y1, row_y1))
            bh       = y2 - y1
            rh       = row_y2 - row_y1
            if bh > 0 and rh > 0 and overlap / min(bh, rh) > 0.4:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    result = []
    for row in rows:
        row.sort(key=lambda b: b[1])
        result.extend(b[0] for b in row)
    return result


def _y_iou(a: list, b: list) -> float:
    inter = max(0.0, min(a[_BOX_YMAX], b[_BOX_YMAX]) - max(a[_BOX_YMIN], b[_BOX_YMIN]))
    union = max(a[_BOX_YMAX], b[_BOX_YMAX]) - min(a[_BOX_YMIN], b[_BOX_YMIN])
    return inter / union if union > 0 else 0.0


def _x_iou(a: list, b: list) -> float:
    inter = max(0.0, min(a[_BOX_XMAX], b[_BOX_XMAX]) - max(a[_BOX_XMIN], b[_BOX_XMIN]))
    union = max(a[_BOX_XMAX], b[_BOX_XMAX]) - min(a[_BOX_XMIN], b[_BOX_XMIN])
    return inter / union if union > 0 else 0.0


def _clean_markdown_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'^[•\-\u2022]\s*', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    df = df.fillna("").astype(str)
    df = df.apply(lambda col: col.str.replace(r"\n+", " ", regex=True).str.strip())
    md = df.to_markdown(index=False)
    if md is None:
        headers    = df.columns.tolist()
        rows       = df.values.tolist()
        col_widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        def fmt_row(cells):
            return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"
        sep  = "| " + " | ".join("-" * w for w in col_widths) + " |"
        md   = "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])
    return md


# ── Table extraction ───────────────────────────────────────────────────
def _assign_spans_to_cells(
    page: fitz.Page,
    pdf_rows: list,
    pdf_cols: list,
    is_header_flags: list,
) -> list[dict]:
    if not pdf_rows or not pdf_cols:
        return []

    margin = CELL_SHRINK_MARGIN_PTS
    cell_grid = []
    for ri, row in enumerate(pdf_rows):
        row_ymin = row[_BOX_YMIN] + margin
        row_ymax = row[_BOX_YMAX] - margin
        row_cells = []
        for col in pdf_cols:
            col_xmin = col[_BOX_XMIN] + margin
            col_xmax = col[_BOX_XMAX] - margin
            if col_xmin >= col_xmax:
                col_xmin, col_xmax = col[_BOX_XMIN], col[_BOX_XMAX]
            if row_ymin >= row_ymax:
                row_ymin, row_ymax = row[_BOX_YMIN], row[_BOX_YMAX]
            row_cells.append({
                "rect":  fitz.Rect(col_xmin, row_ymin, col_xmax, row_ymax),
                "spans": [],
            })
        cell_grid.append(row_cells)

    all_rects  = [c["rect"] for row in cell_grid for c in row]
    table_rect = fitz.Rect(
        min(r.x0 for r in all_rects) - 5,
        min(r.y0 for r in all_rects) - 5,
        max(r.x1 for r in all_rects) + 5,
        max(r.y1 for r in all_rects) + 5,
    )

    text_dict = page.get_text("dict", clip=table_rect)
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                if not span_text:
                    continue
                span_bbox = span.get("bbox")
                if not span_bbox:
                    continue
                span_cx = (span_bbox[0] + span_bbox[2]) / 2
                span_cy = (span_bbox[1] + span_bbox[3]) / 2

                assigned = False
                for ri, row_cells in enumerate(cell_grid):
                    for ci, cell in enumerate(row_cells):
                        r = cell["rect"]
                        if r.x0 <= span_cx <= r.x1 and r.y0 <= span_cy <= r.y1:
                            cell["spans"].append((span_cx, span_cy, span_text))
                            assigned = True
                            break
                    if assigned:
                        break

                if not assigned:
                    min_dist_sq = float("inf")
                    best_ri, best_ci = 0, 0
                    for ri, row_cells in enumerate(cell_grid):
                        for ci, cell in enumerate(row_cells):
                            r = cell["rect"]
                            ccx = (r.x0 + r.x1) / 2
                            ccy = (r.y0 + r.y1) / 2
                            d   = (span_cx - ccx) ** 2 + (span_cy - ccy) ** 2
                            if d < min_dist_sq:
                                min_dist_sq    = d
                                best_ri, best_ci = ri, ci
                    if min_dist_sq <= MAX_NEAREST_CELL_DIST_SQ:
                        cell_grid[best_ri][best_ci]["spans"].append(
                            (span_cx, span_cy, span_text)
                        )

    result = []
    for ri, row_cells in enumerate(cell_grid):
        cells_text = []
        for cell in row_cells:
            cell["spans"].sort(key=lambda s: (s[1], s[0]))
            cells_text.append(" ".join(s[2] for s in cell["spans"]))
        result.append({
            "cells":     cells_text,
            "is_header": is_header_flags[ri] if ri < len(is_header_flags) else False,
        })
    return result


def _extract_table_with_transformer(
    page: fitz.Page,
    img_array: np.ndarray,
    yolo_box: list,
    page_img_width: int,
    page_img_height: int,
) -> pd.DataFrame | None:
    raw_x1, raw_y1, raw_x2, raw_y2 = [int(v) for v in yolo_box]
    h, w = img_array.shape[:2]
    x1 = max(0,     raw_x1 - TABLE_CROP_PADDING)
    y1 = max(0,     raw_y1 - TABLE_CROP_PADDING)
    x2 = min(w - 1, raw_x2 + TABLE_CROP_PADDING)
    y2 = min(h - 1, raw_y2 + TABLE_CROP_PADDING)

    cropped = img_array[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    pil_img     = Image.fromarray(cropped).convert("RGB")
    crop_w, crop_h = pil_img.size
    target_sizes   = torch.tensor([[crop_h, crop_w]])

    inputs = table_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = table_model(**inputs)

    results = table_processor.post_process_object_detection(
        outputs, threshold=TABLE_TRANSFORMER_THRESHOLD, target_sizes=target_sizes,
    )[0]

    raw_rows, raw_cols = [], []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = table_model.config.id2label[label.item()]
        if label_name == "table column header":
            raw_rows.append({"box": box.tolist(), "is_header": True,  "score": score.item()})
        elif label_name == "table row":
            raw_rows.append({"box": box.tolist(), "is_header": False, "score": score.item()})
        elif label_name == "table column":
            raw_cols.append(box.tolist())

    # dedup rows
    kept_rows = []
    for candidate in raw_rows:
        merged = False
        for i, existing in enumerate(kept_rows):
            if _y_iou(candidate["box"], existing["box"]) > TABLE_DEDUP_IOU_THRESHOLD:
                if candidate["is_header"] and not existing["is_header"]:
                    kept_rows[i] = dict(candidate)
                elif candidate["score"] > existing["score"] and candidate["is_header"] == existing["is_header"]:
                    kept_rows[i] = dict(candidate)
                merged = True
                break
        if not merged:
            kept_rows.append(dict(candidate))

    # dedup cols
    kept_cols = []
    for candidate in raw_cols:
        merged = False
        for i, existing in enumerate(kept_cols):
            if _x_iou(candidate, existing) > TABLE_DEDUP_IOU_THRESHOLD:
                if (candidate[_BOX_XMAX] - candidate[_BOX_XMIN]) > (existing[_BOX_XMAX] - existing[_BOX_XMIN]):
                    kept_cols[i] = candidate
                merged = True
                break
        if not merged:
            kept_cols.append(candidate)

    kept_rows.sort(key=lambda r: r["box"][_BOX_YMIN])
    kept_cols.sort(key=lambda c: c[_BOX_XMIN])

    rows            = [r["box"] for r in kept_rows]
    cols            = kept_cols
    is_header_flags = [r["is_header"] for r in kept_rows]

    if not rows or not cols:
        return None

    pdf_w, pdf_h = page.rect.width, page.rect.height
    scale_x = pdf_w / page_img_width
    scale_y = pdf_h / page_img_height

    def crop_to_pdf(box):
        return [
            (box[_BOX_XMIN] + x1) * scale_x,
            (box[_BOX_YMIN] + y1) * scale_y,
            (box[_BOX_XMAX] + x1) * scale_x,
            (box[_BOX_YMAX] + y1) * scale_y,
        ]

    pdf_rows = [crop_to_pdf(r) for r in rows]
    pdf_cols = [crop_to_pdf(c) for c in cols]

    row_dicts = _assign_spans_to_cells(page, pdf_rows, pdf_cols, is_header_flags)
    if not row_dicts:
        return None

    header_indices = [i for i, rd in enumerate(row_dicts) if rd["is_header"]]
    if header_indices:
        header_idx = header_indices[0]
        headers    = row_dicts[header_idx]["cells"]
        data_rows  = [rd["cells"] for i, rd in enumerate(row_dicts) if i != header_idx]
        n_cols     = len(cols)
        if len(headers) < n_cols:
            headers += [f"Col_{j}" for j in range(len(headers), n_cols)]
        elif len(headers) > n_cols:
            headers = headers[:n_cols]
        return pd.DataFrame(data_rows, columns=headers)
    else:
        return pd.DataFrame([rd["cells"] for rd in row_dicts])


# ── Post-processing ────────────────────────────────────────────────────
def _clean_noise(md: str) -> str:
    md = re.sub(r'[\u0900-\u097F]+', '', md)
    md = re.sub(r'(?im)^page\s+\d+\s+of\s+\d+$',  '', md)
    md = re.sub(r'(?im)^\d+\s*\|\s*page$',          '', md)
    md = re.sub(r'(?im)^-\s*\d+\s*-$',              '', md)
    md = re.sub(r'(?im)^(confidential|draft|internal use only|proprietary).*$', '', md)
    md = re.sub(r'(?im)^([A-Z]{4,}\s*){2,}$',       '', md)
    md = re.sub(r'(?m)^\s*\d+\s*$',                 '', md)
    return md


def _postprocess_markdown(md: str) -> str:
    md = _clean_noise(md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ── Metadata helpers ───────────────────────────────────────────────────
def _make_doc_id(file_name: str, file_size_kb: float) -> str:
    return hashlib.md5(f"{file_name}{file_size_kb}".encode()).hexdigest()[:12]


def _extract_filename_tokens(file_name: str) -> list[str]:
    stem   = Path(file_name).stem.lower()
    tokens = re.split(r'[_\-\s]+', stem)
    return [t for t in tokens if t]


def _extract_doc_metadata(file_name: str, markdown: str = "") -> dict:
    meta = {}
    name = Path(file_name).stem

    year = re.search(r'\b(19\d{2}|20\d{2})\b', name)
    if year:
        meta["year"] = year.group(1)
        return meta

    if markdown:
        heading_years = []
        for line in markdown.split("\n"):
            if line.strip().startswith("#"):
                for y in re.findall(r'\b(19\d{2}|20\d{2})\b', line):
                    heading_years.append(int(y))
        if heading_years:
            meta["year"] = str(max(heading_years))
            return meta

    if markdown:
        counts: dict = {}
        for y in re.findall(r'\b(19\d{2}|20\d{2})\b', markdown[:2000]):
            counts[y] = counts.get(y, 0) + 1
        if counts:
            meta["year"] = max(counts, key=lambda k: counts[k])

    return meta


# ── LLM metadata ───────────────────────────────────────────────────────
async def _generate_summary_async(markdown: str, client: httpx.AsyncClient) -> str:
    try:
        headings = [l.strip() for l in markdown.split("\n") if l.strip().startswith("#")][:10]
        mid      = len(markdown) // 2
        body     = markdown[:SUMMARY_MAX_CHARS] + "\n\n" + markdown[mid: mid + 1500]
        prompt   = (
            "You are a document summarizer.\n"
            "Read the document excerpt below.\n"
            "First line: write a short title (max 10 words) describing the document.\n"
            "Then write a 4-5 line summary describing:\n"
            "- What this document is about\n"
            "- Who are the main parties or subjects\n"
            "- What is the key issue or finding\n"
            "- What domain this belongs to\n"
            "Format: Title on first line, then a blank line, then the summary paragraph. No bullets.\n\n"
            f"Headings:\n{chr(10).join(headings)}\n\nContent:\n{body}"
        )
        response = await client.post(
            SUMMARY_OLLAMA_URL,
            json={"model": SUMMARY_MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": 0},
            timeout=120,
        )
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
        logger.info("Summary generated | chars=%d", len(summary))
        return summary
    except Exception as e:
        logger.warning("Summary generation failed | %s", e)
        return ""


async def _extract_keywords_async(markdown: str, client: httpx.AsyncClient) -> list[str]:
    try:
        headings = [l.strip() for l in markdown.split("\n") if l.strip().startswith("#")][:10]
        body     = markdown[:1500]
        prompt   = (
            "You are a keyword extractor.\n"
            "Read the document excerpt below.\n"
            "Return a JSON array of max 10 keywords.\n"
            "Keywords must be:\n"
            "- Important entities, concepts, technologies, laws, organisations, people, places, products, or events\n"
            "- Terms that best identify the document uniquely\n"
            "- Short phrases with maximum 3 words\n"
            "- Mix broad topics and specific identifiers when relevant\n"
            "Avoid generic words like 'document', 'content', 'section', 'chapter', 'introduction'.\n"
            "Return ONLY a JSON array. No explanation. No markdown.\n"
            'Example: ["machine learning", "OpenAI", "financial fraud"]\n\n'
            f"Headings:\n{chr(10).join(headings)}\n\nContent:\n{body}"
        )
        response = await client.post(
            SUMMARY_OLLAMA_URL,
            json={"model": SUMMARY_MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": 0},
            timeout=120,
        )
        response.raise_for_status()
        raw      = re.sub(r"```json|```", "", response.json().get("response", "").strip()).strip()
        keywords = json.loads(raw)
        if isinstance(keywords, list):
            keywords = [k for k in keywords if isinstance(k, str)][:10]
            logger.info("Keywords extracted | count=%d", len(keywords))
            return keywords
        return []
    except Exception as e:
        logger.warning("Keyword extraction failed | %s", e)
        return []


async def _generate_metadata_parallel(markdown: str) -> tuple[str, list[str]]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        summary, keywords = await asyncio.gather(
            _generate_summary_async(markdown, client),
            _extract_keywords_async(markdown, client),
        )
    return summary, keywords


def _run_async_safe(coro):
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def warmup_summary_model() -> None:
    try:
        logger.info("Warming up summary model | %s", SUMMARY_MODEL)
        httpx.post(
            SUMMARY_OLLAMA_URL,
            json={"model": SUMMARY_MODEL, "prompt": "hello", "stream": False, "think": False, "keep_alive": 300},
            timeout=60,
        )
        logger.info("Summary model warm | %s", SUMMARY_MODEL)
    except Exception as e:
        logger.warning("Summary model warmup failed | %s", e)


# ── Main parser ────────────────────────────────────────────────────────
class DocumentParser:

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DocumentParser initialized | output_dir=%s", self.output_dir)

    # ── Public: parse bytes ────────────────────────────────────────────
    def parse_bytes(self, file_bytes: bytes, file_name: str) -> ParseResult:
        suffix       = Path(file_name).suffix.lower()
        file_size_mb = len(file_bytes) / (1024 * 1024)

        if suffix not in SUPPORTED_EXTENSIONS:
            return self._skipped(file_name, suffix, f"Unsupported: {suffix}")
        if file_size_mb > MAX_FILE_SIZE_MB:
            return self._skipped(file_name, suffix, f"Too large: {file_size_mb:.1f}MB")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            return self.parse_file(tmp_path, original_file_name=file_name)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Public: parse file ─────────────────────────────────────────────
    def parse_file(
        self,
        file_path:          str | Path,
        original_file_name: str = None,
    ) -> ParseResult:
        file_path = Path(file_path)
        start     = time.perf_counter()

        if not file_path.exists() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return self._skipped(file_path.name, file_path.suffix, f"Missing or unsupported: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        file_size_kb = file_path.stat().st_size / 1024

        if file_size_mb > MAX_FILE_SIZE_MB:
            return self._skipped(file_path.name, file_path.suffix, f"Too large: {file_size_mb:.1f}MB")

        name_for_meta   = original_file_name or file_path.name
        doc_id          = _make_doc_id(name_for_meta, round(file_size_kb, 2))
        filename_tokens = _extract_filename_tokens(name_for_meta)

        logger.info("Parsing | %s | %.1f MB", name_for_meta, file_size_mb)

        warnings: list[str]        = []
        blocks:   list[BlockRecord] = []
        pages:    list[PageRecord]  = []
        markdown_lines: list[str]  = []

        try:
            doc = fitz.open(str(file_path))
            total_pages = len(doc)

            current_section = ""
            for page_num in range(total_pages):
                page = doc[page_num]
                page_no    = page_num + 1
                page_label = str(page_no)

                page_blocks, page_md, current_section = self._process_page(
                    page, page_no, page_label, warnings, current_section
                )
                blocks.extend(page_blocks)
                markdown_lines.extend(page_md)

                page_text = " ".join(
                    b.content for b in page_blocks if not b.is_table
                )
                pages.append(PageRecord(
                    page_no    = page_no,
                    page_label = page_label,
                    text       = page_text.strip(),
                ))

            doc.close()

        except Exception as e:
            logger.error("Parse failed | %s | %s", name_for_meta, e)
            return ParseResult(
                file_name           = name_for_meta,
                file_type           = file_path.suffix,
                method_used         = ExtractionMethod.ERROR,
                markdown            = "",
                success             = False,
                error               = str(e),
                warnings            = warnings,
                processing_time_sec = round(time.perf_counter() - start, 3),
            )

        raw_markdown = _postprocess_markdown("\n\n".join(markdown_lines))
        self._save(file_path, raw_markdown, name_for_meta)

        elapsed  = round(time.perf_counter() - start, 3)
        doc_meta = _extract_doc_metadata(name_for_meta, raw_markdown)
        doc_year = doc_meta.get("year", "")
        has_tables = any(b.is_table for b in blocks)

        summary, keywords = _run_async_safe(_generate_metadata_parallel(raw_markdown))

        meta = DocumentMeta(
            doc_id          = doc_id,
            file_name       = name_for_meta,
            file_path       = str(file_path),
            file_type       = file_path.suffix,
            file_size_kb    = round(file_size_kb, 2),
            page_count      = len(pages),
            has_tables      = has_tables,
            parse_success   = True,
            filename_tokens = filename_tokens,
            doc_year        = doc_year,
            summary         = summary,
            keywords        = keywords,
            warnings        = warnings,
        )

        logger.info(
            "Done | %s | pages=%d | blocks=%d | tables=%d | time=%.3fs",
            name_for_meta, len(pages), len(blocks),
            sum(1 for b in blocks if b.is_table), elapsed,
        )

        return ParseResult(
            file_name           = name_for_meta,
            file_type           = file_path.suffix,
            method_used         = ExtractionMethod.YOLO,
            markdown            = raw_markdown,
            meta                = meta,
            total_pages         = total_pages,
            success             = True,
            warnings            = warnings,
            pages               = pages,
            blocks              = blocks,
            processing_time_sec = elapsed,
            doc_id              = doc_id,
            doc_year            = doc_year,
            filename_tokens     = filename_tokens,
            summary             = summary,
            keywords            = keywords,
        )

    # ── Per-page processing ────────────────────────────────────────────
    def _process_page(
        self,
        page:       fitz.Page,
        page_no:    int,
        page_label: str,
        warnings:   list,
        current_section: str,
    ) -> tuple[list[BlockRecord], list[str], str]:
        """
        Returns (blocks, markdown_lines) for one page.
        Pages that are scanned/complex/blank yield no blocks and a note in markdown.
        """
        blocks: list[BlockRecord] = []
        md:     list[str]         = []

        raw_page_text = page.get_text("text").strip()
        has_text      = len(raw_page_text) > 0
        has_images    = len(page.get_images()) > 0

        pix       = page.get_pixmap(dpi=150)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        det_res = yolo_model.predict(
            img_array, imgsz=1024, conf=0.2, device=DEVICE, verbose=False
        )
        yolo_boxes = det_res[0].boxes

        element_counts = Counter()
        for b in yolo_boxes:
            lbl = yolo_model.names[int(b.cls[0].item())].lower()
            if lbl != "abandon":
                element_counts[lbl] += 1

        total_elements = sum(element_counts.values())
        text_count     = element_counts.get("plain text", 0) + element_counts.get("title", 0)

        # ── Route: complex / scanned / blank ──────────────────────────
        if text_count >= COMPLEX_TEXT_THRESHOLD or total_elements >= COMPLEX_TOTAL_THRESHOLD:
            logger.info("Page %d: complex — skipping", page_no)
            warnings.append(f"Page {page_no}: complex layout, skipped")
            md.append(f"## Page {page_no}\n\n> [Complex page — skipped]\n")
            return blocks, md, current_section

        if not has_text and not has_images and not total_elements:
            logger.info("Page %d: blank", page_no)
            return blocks, md, current_section

        if not has_text and (has_images or total_elements):
            logger.info("Page %d: scanned/no-text — skipping", page_no)
            warnings.append(f"Page {page_no}: scanned, no text layer")
            md.append(f"## Page {page_no}\n\n> [Scanned page — no text layer]\n")
            return blocks, md, current_section

        # ── Build reading order ────────────────────────────────────────
        boxes_data = []
        for idx in range(len(yolo_boxes)):
            b = yolo_boxes[idx]
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            boxes_data.append((idx, bx1, by1, bx2, by2))

        sorted_indices = _get_dynamic_reading_order(boxes_data)

        md.append(f"## Page {page_no}\n")
        seen_texts: set[str] = set()
        current_section      = ""

        text_labels   = {"title", "plain text", "figure_caption",
                         "table_caption", "table_footnote", "formula_caption"}

        for idx in sorted_indices:
            b      = yolo_boxes[idx]
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            label  = yolo_model.names[cls_id].lower()

            if label == "abandon":
                continue

            scale_x = page.rect.width  / pix.width
            scale_y = page.rect.height / pix.height
            rect    = fitz.Rect(bx1 * scale_x, by1 * scale_y, bx2 * scale_x, by2 * scale_y)

            # ── Text elements ──────────────────────────────────────────
            if label in text_labels:
                raw_text = page.get_text("text", clip=rect, sort=True).strip()
                if not raw_text:
                    continue
                text_hash = re.sub(r'\s+', '', raw_text.lower())
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)
                clean = _clean_markdown_text(raw_text)

                if label == "title":
                    current_section = clean
                    md.append(f"### {clean}\n")
                    blocks.append(BlockRecord(
                        block_type = "heading",
                        content    = clean,
                        page_no    = page_no,
                        page_label = page_label,
                        section    = current_section,
                        is_table   = False,
                    ))

                elif label == "plain text":
                    md.append(f"{clean}\n")
                    blocks.append(BlockRecord(
                        block_type = "text",
                        content    = clean,
                        page_no    = page_no,
                        page_label = page_label,
                        section    = current_section,
                        is_table   = False,
                    ))

                else:
                    # captions / footnotes → text block, not heading
                    md.append(f"*{clean}*\n")
                    blocks.append(BlockRecord(
                        block_type = "text",
                        content    = clean,
                        page_no    = page_no,
                        page_label = page_label,
                        section    = current_section,
                        is_table   = False,
                    ))

            # ── Table ──────────────────────────────────────────────────
            elif label == "table":
                df = _extract_table_with_transformer(
                    page           = page,
                    img_array      = img_array,
                    yolo_box       = b.xyxy[0].tolist(),
                    page_img_width = pix.width,
                    page_img_height= pix.height,
                )
                if df is not None and not df.empty:
                    md_table = _dataframe_to_markdown(df)
                    md.append(md_table + "\n")
                    blocks.append(BlockRecord(
                        block_type = "table",
                        content    = md_table,
                        page_no    = page_no,
                        page_label = page_label,
                        section    = current_section,
                        is_table   = True,
                    ))
                else:
                    logger.warning("Page %d: table extraction failed", page_no)
                    warnings.append(f"Page {page_no}: table grid detection failed")

        return blocks, md, current_section

    # ── Folder parsing ─────────────────────────────────────────────────
    def parse_folder(
        self,
        folder_path: str | Path,
        recursive:   bool = False,
        max_workers: int  = 4,
    ) -> list[ParseResult]:
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")

        pattern = "**/*" if recursive else "*"
        files   = [
            f for f in folder_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        logger.info("Parsing %d file(s) | workers=%d", len(files), max_workers)
        warmup_summary_model()

        if max_workers == 1:
            results = [self.parse_file(f) for f in files]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._parse_file_wrapper, f): f for f in files}
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        results.append(future.result(timeout=PARSE_TIMEOUT_SEC))
                    except Exception as e:
                        logger.error("Failed | %s | %s", fp, e)
                        results.append(self._skipped(fp.name, fp.suffix, str(e)))

        success = sum(1 for r in results if r.success)
        logger.info(
            "Batch done | total=%d | ok=%d | failed=%d",
            len(results), success, len(results) - success,
        )
        return results

    def _parse_file_wrapper(self, file_path: Path) -> ParseResult:
        try:
            return self.parse_file(file_path)
        except Exception as e:
            return self._skipped(file_path.name, file_path.suffix, str(e))

    # ── Helpers ────────────────────────────────────────────────────────
    def _save(self, file_path: Path, markdown: str, original_name: str = None) -> None:
        if self.output_dir and markdown.strip():
            stem = Path(original_name).stem if original_name else file_path.stem
            (self.output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")   

    def _skipped(self, file_name: str, file_type: str, reason: str) -> ParseResult:
        logger.warning("Skipped | %s | %s", file_name, reason)
        return ParseResult(
            file_name   = file_name,
            file_type   = file_type,
            method_used = ExtractionMethod.SKIPPED,
            markdown    = "",
            success     = False,
            error       = reason,
        )