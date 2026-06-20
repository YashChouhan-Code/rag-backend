"""
agent/assembler.py

Assembler — third stage in the new RAG pipeline.
Pipeline: QueryCleaner → Retriever → Assembler → Orchestrator → LLM

Adapts context block structure based on answer_structure signal:
    direct     → flat best-first ordering
    compare    → grouped by arm_label with equal sections
    synthesize → doc summary + representative chunks per doc
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict

from final_rag.agent.models import SourceInfo, AssembledResult, CleanedQuery
from final_rag.agent.retriever import RetrievedChunk
import final_rag.config as config

logger = logging.getLogger("agent.assembler")


class Assembler:
    """
    Assembles context_block from CleanedQuery + list[RetrievedChunk].
    Structure adapts to answer_structure signal from QueryCleaner.
    """

    def __init__(self, max_context_tokens: int = config.MAX_CONTEXT_TOKENS):
        self.max_context_tokens = max_context_tokens
        logger.info("Assembler initialized | token_limit=%d", max_context_tokens)

    # ── Main entry ─────────────────────────────────────────────────────
    def assemble(
        self,
        cleaned: CleanedQuery,
        chunks:  list[RetrievedChunk],
    ) -> AssembledResult:

        start = time.perf_counter()

        if not chunks:
            logger.warning("[Assembler] No chunks received.")
            return self._not_found_result(start, cleaned.answer_structure)

        logger.info(
            "[Assembler] Received %d chunks | structure=%s | query='%.60s'",
            len(chunks), cleaned.answer_structure, cleaned.improved_query,
        )

        # ── Step 1: Deduplicate ────────────────────────────────────────
        chunks = self._deduplicate(chunks)

        # ── Step 2: Prioritize ─────────────────────────────────────────
        chunks = self._prioritize(chunks)

        # ── Step 3: Token budget ───────────────────────────────────────
        chunks, was_trimmed = self._apply_token_budget(chunks, cleaned.answer_structure)

        # ── Step 4: Build context block (structure-aware) ──────────────
        context_block = self._build_context_block(chunks, cleaned.answer_structure)

        # ── Step 5: Build sources ──────────────────────────────────────
        sources = self._build_sources(chunks)

        elapsed = round(time.perf_counter() - start, 3)
        logger.info(
            "[Assembler] Done | chunks=%d | sources=%d | tokens~=%d | "
            "trimmed=%s | time=%.3fs",
            len(chunks), len(sources),
            self._estimate_tokens(context_block),
            was_trimmed, elapsed,
        )

        return AssembledResult(
            context_block       = context_block,
            sources             = sources,
            chunks_used         = len(chunks),
            has_weak_match      = any(c.is_weak_match for c in chunks),
            has_tables          = any(c.is_table for c in chunks),
            not_found           = False,
            was_trimmed         = was_trimmed,
            processing_time_sec = elapsed,
            answer_structure    = cleaned.answer_structure,
            sources_count       = len(sources),
        )

    # ── Step 1: Deduplicate ────────────────────────────────────────────
    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen   = set()
        unique = []
        for chunk in chunks:
            fp = hashlib.md5(chunk.text.strip().lower().encode()).hexdigest()
            if fp not in seen:
                seen.add(fp)
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        if removed:
            logger.info("[Assembler] Dedup removed %d duplicate chunks.", removed)
        return unique

    # ── Step 2: Prioritize ─────────────────────────────────────────────
    def _prioritize(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Tables first → normal by rerank_score → expanded last.
        Preserves arm_label grouping for compare structure.
        """
        tables   = [c for c in chunks if c.is_table and not c.is_temporal_expanded]
        normal   = [c for c in chunks if not c.is_table and not c.is_temporal_expanded]
        expanded = [c for c in chunks if c.is_temporal_expanded]

        # sort normal by rerank score, but keep arm_label groups together
        normal.sort(key=lambda c: (c.arm_label, -c.rerank_score))

        return tables + normal + expanded

    # ── Step 3: Token budget ───────────────────────────────────────────
    def _apply_token_budget(
        self,
        chunks:           list[RetrievedChunk],
        answer_structure: str,
    ) -> tuple[list[RetrievedChunk], bool]:
        """
        For compare: allocate equal budget per arm.
        For direct/synthesize: fill budget best-first.
        Tables always force in.
        """
        if answer_structure == "compare":
            return self._budget_compare(chunks)
        else:
            return self._budget_simple(chunks)

    def _budget_simple(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool]:
        selected    = []
        used        = 0
        was_trimmed = False

        for chunk in chunks:
            tokens = self._estimate_tokens(chunk.text)

            if used + tokens <= self.max_context_tokens:
                selected.append(chunk)
                used += tokens
            else:
                # tables force in despite budget
                if chunk.is_table:
                    selected.append(chunk)
                    used += tokens
                    logger.debug("[Assembler] Table forced in despite budget.")
                else:
                    was_trimmed = True
                    logger.info(
                        "[Assembler] Budget hit at ~%d tokens. Dropping %d chunks.",
                        used, len(chunks) - len(selected),
                    )
                    break

        return selected, was_trimmed

    def _budget_compare(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool]:
        """
        Equal budget per arm. If one arm exhausts budget, others still get their share.
        """
        arm_groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for c in chunks:
            arm_groups[c.arm_label or "default"].append(c)

        num_arms       = len(arm_groups)
        budget_per_arm = self.max_context_tokens // num_arms if num_arms > 0 else self.max_context_tokens
        selected       = []
        was_trimmed    = False

        for arm_label, arm_chunks in arm_groups.items():
            arm_used = 0
            for chunk in arm_chunks:
                tokens = self._estimate_tokens(chunk.text)
                if arm_used + tokens <= budget_per_arm:
                    selected.append(chunk)
                    arm_used += tokens
                else:
                    if chunk.is_table:
                        selected.append(chunk)
                        arm_used += tokens
                    else:
                        was_trimmed = True
                        break

            logger.info(
                "[Assembler] Arm '%s' | chunks=%d | tokens~=%d",
                arm_label, len([c for c in selected if c.arm_label == arm_label]), arm_used,
            )

        return selected, was_trimmed

    # ── Step 4: Build context block ────────────────────────────────────
    def _build_context_block(
        self, chunks: list[RetrievedChunk], answer_structure: str
    ) -> str:
        if answer_structure == "compare":
            return self._build_compare_block(chunks)
        elif answer_structure == "synthesize":
            return self._build_synthesize_block(chunks)
        else:
            return self._build_direct_block(chunks)

    # ── direct: flat best-first ordering ───────────────────────────────
    def _build_direct_block(self, chunks: list[RetrievedChunk]) -> str:
        """Current default behavior — flat list grouped by document."""
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.source_file].append(chunk)

        lines = ["=== KNOWLEDGE BASE CONTEXT ==="]

        for file_name, file_chunks in grouped.items():
            sample   = file_chunks[0]
            doc_meta = " ".join(filter(None, [
                sample.doc_org.upper() if sample.doc_org else "",
                sample.doc_year,
            ]))
            doc_header = f"{file_name} ({doc_meta})" if doc_meta else file_name
            lines.append(f"\n--- DOCUMENT: {doc_header} ---")

            for chunk in file_chunks:
                lines.append(self._format_chunk(chunk))

        lines.append("=== END OF CONTEXT ===")
        return "\n".join(lines)

    # ── compare: grouped by arm_label ──────────────────────────────────
    def _build_compare_block(self, chunks: list[RetrievedChunk]) -> str:
        """One section per comparison arm. LLM synthesizes parallel structure."""
        arm_groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            arm_groups[chunk.arm_label or "General"].append(chunk)

        lines = ["=== COMPARISON CONTEXT ==="]
        lines.append("Each section below represents one of the documents or entities being compared.\n")

        for arm_label, arm_chunks in arm_groups.items():
            sample = arm_chunks[0]
            lines.append(f"\n━━━ {arm_label.upper()} ━━━")
            lines.append(f"Source: {sample.source_file}")
            if sample.doc_year:
                lines.append(f"Year: {sample.doc_year}")
            lines.append("")

            for chunk in arm_chunks:
                lines.append(self._format_chunk(chunk, show_file=False))

        lines.append("=== END OF COMPARISON CONTEXT ===")
        return "\n".join(lines)

    # ── synthesize: doc summary + chunks ───────────────────────────────
    def _build_synthesize_block(self, chunks: list[RetrievedChunk]) -> str:
        """Doc summary first, then representative chunks. LLM aggregates."""
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            key = chunk.doc_id if chunk.doc_id else chunk.source_file
            grouped[key].append(chunk)

        lines = ["=== SYNTHESIZED CONTEXT FROM MULTIPLE DOCUMENTS ==="]
        lines.append("Below are summaries and key excerpts from relevant documents.\n")

        for doc_key, doc_chunks in grouped.items():
            sample  = doc_chunks[0]
            summary = sample.summary.strip() if sample.summary else None

            doc_header = sample.source_file
            if sample.doc_year:
                doc_header += f" ({sample.doc_year})"

            lines.append(f"\n━━━ {doc_header} ━━━")

            # doc summary first if available
            if summary:
                lines.append("[DOCUMENT SUMMARY]")
                lines.append(summary)
                lines.append("")

            # then best 1-2 chunks as evidence
            for chunk in doc_chunks[:2]:
                lines.append(self._format_chunk(chunk, show_file=False))

        lines.append("=== END OF SYNTHESIZED CONTEXT ===")
        return "\n".join(lines)

    # ── Chunk formatter ────────────────────────────────────────────────
    def _format_chunk(
        self, chunk: RetrievedChunk, show_file: bool = True
    ) -> str:
        """Single chunk formatted with metadata labels."""
        page_uncertain = "page_no_inference_failed" in chunk.warnings

        if chunk.page_label:
            display_page = chunk.page_label
        elif chunk.page_no > 0:
            display_page = str(chunk.page_no)
        else:
            display_page = "Unknown"

        if page_uncertain and display_page != "Unknown":
            display_page = f"~{display_page}"

        # chunk type
        if chunk.is_table:
            data_type = "[TABLE]"
        elif chunk.is_temporal_expanded:
            data_type = "[EXPANDED]"
        else:
            data_type = "[TEXT]"

        confidence = " [LOW CONFIDENCE]" if chunk.is_weak_match else ""
        section    = chunk.section or "General"

        parts = []
        if show_file:
            parts.append(f"File: {chunk.source_file}")
        parts.append(f"Page {display_page} | {section} {data_type}{confidence}")

        header = " | ".join(parts)
        return f"{header}\n{chunk.text.strip()}\n"

    # ── Step 5: Build sources ──────────────────────────────────────────
    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[SourceInfo]:
        """Citation metadata for API response."""
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            if not chunk.is_temporal_expanded:
                grouped[chunk.source_file].append(chunk)

        sources = []
        for file_name, file_chunks in grouped.items():
            pages_set: set[str] = set()
            for c in file_chunks:
                page_uncertain = "page_no_inference_failed" in c.warnings
                if c.page_label:
                    label = f"~{c.page_label}" if page_uncertain else str(c.page_label)
                    pages_set.add(label)
                elif c.page_no > 0:
                    label = f"~{c.page_no}" if page_uncertain else str(c.page_no)
                    pages_set.add(label)

            pages = sorted(
                pages_set,
                key=lambda x: int(x.lstrip("~")) if x.lstrip("~").isdigit() else float("inf"),
            )
            sources.append(SourceInfo(
                file_name   = file_name,
                pages       = pages,
                chunk_count = len(file_chunks),
            ))

        return sources

    # ── Fallback result ────────────────────────────────────────────────
    def _not_found_result(self, start: float, answer_structure: str) -> AssembledResult:
        return AssembledResult(
            context_block       = "No relevant matches found in the knowledge base.",
            sources             = [],
            chunks_used         = 0,
            has_weak_match      = False,
            has_tables          = False,
            not_found           = True,
            was_trimmed         = False,
            processing_time_sec = round(time.perf_counter() - start, 3),
            answer_structure    = answer_structure,
            sources_count       = 0,
        )

    # ── Utility ────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * config.TOKENS_PER_WORD)


def get_assembler() -> Assembler:
    return Assembler()