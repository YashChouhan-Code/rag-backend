"""
ingestion/chunker.py

Chunker — consumes structured BlockRecord list from parser.
Table detection comes from parser metadata (no regex detection).
Stack: SentenceSplitter + tiktoken
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from final_rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHARS,
    MIN_WORDS,
    TABLE_MAX_TOKENS,
    DEDUP_THRESHOLD,
    DEDUP_SHINGLE_SIZE,
)
from final_rag.ingestion.parser import BlockRecord, PageRecord, ParseResult

logger = logging.getLogger("ingestion.chunker")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    chunk_id:        str
    chunk_index:     int
    text:            str
    page_no:         int       = 0
    page_label:      str       = ""
    page_range: list[int] = field(default_factory=lambda: [0, 0])
    section:         str       = ""
    is_table:        bool      = False
    token_count:     int       = 0
    chunk_year:      list[str] = field(default_factory=list)
    keywords:        list[str] = field(default_factory=list)
    doc_id:          str       = ""
    source_file:     str       = ""
    filename_tokens: list[str] = field(default_factory=list)
    doc_year:        str       = ""
    has_tables:      bool      = False
    summary:         str       = ""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _load_tokenizer():
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        logger.info("tiktoken loaded — cl100k_base")
        return encoder
    except Exception as exc:
        logger.warning("tiktoken unavailable, falling back to word count | %s", exc)
        return None


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# chunk_id generator
# ---------------------------------------------------------------------------

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _sanitize_whitespace(text: str) -> str:
    cleaned_lines = []
    for line in text.split("\n"):
        if "|" in line:
            cleaned_lines.append(line.strip())
        else:
            cleaned_lines.append(re.sub(r" {3,}", " ", line).strip())
    return "\n".join(cleaned_lines).strip()


def _extract_years(text: str) -> list[str]:
    return list(set(re.findall(r'\b(19\d{2}|20\d{2})\b', text)))


def _is_embeddable(text: str, is_table: bool = False) -> bool:
    if is_table:
        return True

    stripped = text.strip()
    non_heading_lines = [
        line.strip()
        for line in stripped.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    if not non_heading_lines:
        return False

    FACTUAL_PATTERN = re.compile(
        r'\d+|rs\.?|inr|₹|crore|lakh|\b(19|20)\d{2}\b',
        re.IGNORECASE,
    )
    has_factual_content = bool(FACTUAL_PATTERN.search(stripped))

    if has_factual_content:
        return len(stripped) >= 20

    return len(stripped.split()) >= MIN_WORDS and len(stripped) >= MIN_CHARS


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _table_to_sentences(table_markdown: str) -> str:
    lines = [
        line.strip()
        for line in table_markdown.strip().split("\n")
        if line.strip() and not re.match(r"^\|[-| :]+\|$", line.strip())
    ]

    if len(lines) < 2:
        return table_markdown

    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    sentences = []
    dropped_rows = 0

    for row_line in lines[1:]:
        cells = [cell.strip() for cell in row_line.strip("|").split("|")]
        if len(cells) != len(headers):
            dropped_rows += 1
            continue
        pairs = ", ".join(
            f"{header}={cell}"
            for header, cell in zip(headers, cells)
            if cell
        )
        if pairs:
            sentences.append(pairs)

    if dropped_rows:
        logger.warning(
            "_table_to_sentences: dropped %d row(s) due to column mismatch "
            "(expected %d columns) — likely OCR noise or merged cells",
            dropped_rows, len(headers),
        )

    return "\n".join(sentences) if sentences else table_markdown


# ---------------------------------------------------------------------------
# Entity fingerprint for smart dedup
# ---------------------------------------------------------------------------

def _extract_entity_fingerprint(text: str) -> str:
    numbers  = re.findall(r'\d+', text)
    years    = re.findall(r'\b(19|20)\d{2}\b', text)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    signature = "|".join(sorted(set(numbers + years + entities[:5])))
    return hashlib.md5(signature.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_exact(chunks: list[ChunkResult]) -> list[ChunkResult]:
    seen_hashes: set[str] = set()
    unique_chunks: list[ChunkResult] = []

    for chunk in chunks:
        normalised = re.sub(r"\s+", " ", chunk.text.strip().lower())
        md5_hash   = hashlib.md5(normalised.encode()).hexdigest()

        if md5_hash not in seen_hashes:
            seen_hashes.add(md5_hash)
            unique_chunks.append(chunk)
        else:
            logger.debug(
                "Exact dedup dropped | file=%s | index=%d",
                chunk.source_file, chunk.chunk_index,
            )

    return unique_chunks


def _get_shingles(text: str, size: int = DEDUP_SHINGLE_SIZE) -> set[str]:
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return {normalised[i:i + size] for i in range(len(normalised) - size + 1)}


def _dedup_near_smart(
    chunks: list[ChunkResult],
    threshold: float = DEDUP_THRESHOLD,
) -> list[ChunkResult]:
    kept_chunks: list[ChunkResult]    = []
    kept_shingle_sets: list[set[str]] = []
    kept_fingerprints: list[str]      = []

    for chunk in chunks:
        if chunk.is_table:
            kept_chunks.append(chunk)
            continue

        shingle_size          = 12 if len(chunk.text) > 500 else DEDUP_SHINGLE_SIZE
        candidate_shingles    = _get_shingles(chunk.text, shingle_size)
        candidate_fingerprint = _extract_entity_fingerprint(chunk.text)

        is_near_duplicate = False

        for i, existing_shingles in enumerate(kept_shingle_sets):
            if candidate_fingerprint != kept_fingerprints[i]:
                continue

            union_size = len(candidate_shingles | existing_shingles)
            if union_size == 0:
                continue
            jaccard = len(candidate_shingles & existing_shingles) / union_size
            if jaccard >= threshold:
                is_near_duplicate = True
                logger.debug(
                    "Near dedup dropped | file=%s | index=%d | jaccard=%.2f",
                    chunk.source_file, chunk.chunk_index, jaccard,
                )
                break

        if not is_near_duplicate:
            kept_chunks.append(chunk)
            kept_shingle_sets.append(candidate_shingles)
            kept_fingerprints.append(candidate_fingerprint)

    return kept_chunks


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _dominant_section(blocks: list[BlockRecord]) -> str:
    section_names = [block.section for block in blocks if block.section]
    if not section_names:
        return ""
    return Counter(section_names).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class DocumentChunker:

    def __init__(
        self,
        chunk_size: int    = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer     = _load_tokenizer()

        self._splitter = SentenceSplitter(
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            tokenizer     = self.tokenizer.encode if self.tokenizer else None,
        )

        logger.info(
            "DocumentChunker initialised | size=%d | overlap=%d",
            chunk_size, chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, parse_result: ParseResult) -> list[ChunkResult]:
        if not parse_result.success:
            logger.warning("Skipping failed ParseResult | %s", parse_result.file_name)
            return []

        if not parse_result.blocks:
            logger.warning(
                "No blocks in ParseResult | %s — nothing to chunk",
                parse_result.file_name,
            )
            return []

        logger.info(
            "Chunking | file=%s | blocks=%d",
            parse_result.file_name, len(parse_result.blocks),
        )

        chunks = self._process_blocks(
            blocks          = parse_result.blocks,
            page_records    = parse_result.pages,
            doc_id          = parse_result.doc_id,
            source_file     = parse_result.file_name,
            filename_tokens = parse_result.filename_tokens,
            doc_year        = parse_result.doc_year,
            has_tables      = parse_result.meta.has_tables if parse_result.meta else False,
            summary         = parse_result.summary,
            keywords        = parse_result.keywords,
        )

        chunks = self._deduplicate(chunks, parse_result.file_name)
        self._log_summary(chunks, parse_result.file_name)
        return chunks

    def chunk_batch(
        self, parse_results: list[ParseResult]
    ) -> dict[str, list[ChunkResult]]:
        results     = {pr.file_name: self.chunk(pr) for pr in parse_results}
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(
            "Batch done | files=%d | total_chunks=%d",
            len(results), total_chunks,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: list[ChunkResult], file_name: str) -> list[ChunkResult]:
        before  = len(chunks)
        chunks  = _dedup_exact(chunks)
        chunks  = _dedup_near_smart(chunks)
        removed = before - len(chunks)
        if removed:
            logger.info(
                "Dedup removed %d chunks | file=%s | before=%d after=%d",
                removed, file_name, before, len(chunks),
            )
        return chunks

    def _log_summary(self, chunks: list[ChunkResult], file_name: str) -> None:
        logger.info(
            "Done | %s | chunks=%d | tables=%d",
            file_name,
            len(chunks),
            sum(1 for c in chunks if c.is_table),
        )

    def _split_text(self, text: str) -> list[str]:
        try:
            nodes = self._splitter.get_nodes_from_documents([Document(text=text)])
        except Exception as exc:
            logger.warning("SentenceSplitter failed: %s", exc)
            return [text]

        return [
            (node.get_content() if hasattr(node, "get_content") else node.text).strip()
            for node in nodes
        ]

    # ------------------------------------------------------------------
    # Core block processing
    # ------------------------------------------------------------------

    def _process_blocks(
        self,
        blocks:          list[BlockRecord],
        page_records:    list[PageRecord],
        doc_id:          str       = "",
        source_file:     str       = "",
        filename_tokens: list      = None,
        doc_year:        str       = "",
        has_tables:      bool      = False,
        summary:         str       = "",
        keywords:        list      = None,
    ) -> list[ChunkResult]:

        chunks: list[ChunkResult] = []
        chunk_index               = 0

        text_buffer: list[BlockRecord] = []
        buffer_token_count             = 0

        # ------------------------------------------------------------------
        # flush_buffer
        # ------------------------------------------------------------------
        def flush_buffer() -> None:
            nonlocal chunk_index

            if not text_buffer:
                return

            merged_text = "\n".join(
                _sanitize_whitespace(block.content) for block in text_buffer
            )

            if not _is_embeddable(merged_text):
                text_buffer.clear()
                return

            section = _dominant_section(text_buffer)

            # CHANGE 1: read page directly from buffer blocks, no string inference
            start_page = text_buffer[0].page_no
            end_page   = text_buffer[-1].page_no
            page_label = text_buffer[0].page_label

            for split_text in self._split_text(merged_text):
                if not _is_embeddable(split_text):
                    continue

                display_text = f"[{section}]\n{split_text}" if section else split_text

                chunks.append(ChunkResult(
                    chunk_id        = _make_chunk_id(doc_id, chunk_index),
                    chunk_index     = chunk_index,
                    text            = display_text,
                    page_no         = start_page,
                    page_label      = page_label,
                    page_range      = [start_page, end_page],
                    section         = section,
                    is_table        = False,
                    token_count     = _count_tokens(display_text, self.tokenizer),
                    chunk_year      = _extract_years(split_text),
                    doc_id          = doc_id,
                    source_file     = source_file,
                    filename_tokens = filename_tokens or [],
                    doc_year        = doc_year,
                    has_tables      = has_tables,
                    summary         = summary,
                    keywords        = keywords or [],
                ))
                chunk_index += 1

            text_buffer.clear()

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        for block_index, block in enumerate(blocks):
            content = _sanitize_whitespace(block.content)

            # CHANGE 2: heading prepended to buffer instead of dropped
            if block.block_type in ("heading", "title"):
                flush_buffer()
                text_buffer.append(block)
                continue

            # table block
            if block.is_table:
                flush_buffer()

                start_page = block.page_no
                end_page   = block.page_no

                preceding_block = blocks[block_index - 1] if block_index > 0 else None
                bonded_context  = ""
                if (
                    preceding_block
                    and preceding_block.block_type == "text"
                    and preceding_block.content.strip()
                ):
                    bonded_context = _sanitize_whitespace(preceding_block.content)

                if _count_tokens(content, self.tokenizer) > TABLE_MAX_TOKENS:
                    content = _table_to_sentences(content)

                full_text = (
                    f"{bonded_context}\n{content}" if bonded_context else content
                ).strip()

                if not _is_embeddable(full_text, is_table=True):
                    continue

                display_text = (
                    f"[{block.section}]\n{full_text}" if block.section else full_text
                )

                chunks.append(ChunkResult(
                    chunk_id        = _make_chunk_id(doc_id, chunk_index),
                    chunk_index     = chunk_index,
                    text            = display_text,
                    page_no         = start_page,
                    page_label      = block.page_label,
                    page_range      = [start_page, end_page],
                    section         = block.section,
                    is_table        = True,
                    token_count     = _count_tokens(display_text, self.tokenizer),
                    chunk_year      = _extract_years(full_text),
                    doc_id          = doc_id,
                    source_file     = source_file,
                    filename_tokens = filename_tokens or [],
                    doc_year        = doc_year,
                    has_tables      = has_tables,
                    summary         = summary,
                    keywords        = keywords or [],
                ))
                chunk_index += 1
                continue

            # text block
            if not _is_embeddable(content):
                continue

            section_changed = (
                text_buffer and block.section != text_buffer[-1].section
            )
            if section_changed:
                flush_buffer()
                buffer_token_count = 0

            text_buffer.append(block)
            buffer_token_count += _count_tokens(content, self.tokenizer)

            if buffer_token_count >= self.chunk_size:
                flush_buffer()
                buffer_token_count = 0

        flush_buffer()

        return chunks