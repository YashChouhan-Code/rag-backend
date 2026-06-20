"""
agent/retriever.py
Hybrid retriever for RAG pipeline.
OllamaEmbedder (Qwen3 dense + BM25 sparse) + Qwen3 Reranker via Sentence Transformers.

Strategy branches on CleanedQuery signals:
    scope=single → filtered search + filter relaxation fallback
    scope=few    → per-arm filtered search (max 4 arms) + arm tagging
    scope=broad  → unfiltered diversity-capped search
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
import torch
from sentence_transformers import CrossEncoder

from final_rag.agent.models import CleanedQuery, ComparisonArm, RetrievedChunk
from final_rag.ingestion.embedder import OllamaEmbedder
from final_rag.qdrant_storage.store import QdrantManager
import final_rag.config as config

logger = logging.getLogger("agent.retriever")

class Retriever:
    def __init__(self, embedder: OllamaEmbedder, db: QdrantManager):
        self.embedder     = embedder
        self.db           = db
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rerank_model = CrossEncoder(config.RERANKER_MODEL, device=device, trust_remote_code=True)
        logger.info(
            "Retriever initialized | embedder=OllamaEmbedder | reranker=%s on %s",
            config.RERANKER_MODEL,
            device,
        )

    # ── Public entry point ─────────────────────────────────────────────
    def retrieve(self, cleaned: CleanedQuery) -> list[RetrievedChunk]:
        """
        Main retrieval entry point.
        Branches on target_scope from CleanedQuery signals.
        Always uses improved_query + subqueries for embedding.
        improved_query is what LLM will use to answer.
        subqueries are only for finding chunks in Qdrant.
        """
        scope = cleaned.target_scope  # "single" / "few" / "broad"

        logger.info(
            "Retrieval start | scope=%s | structure=%s | specificity=%s",
            scope, cleaned.answer_structure, cleaned.specificity,
        )

        if scope == "single":
            chunks = self._retrieve_single(cleaned)
        elif scope == "few":
            chunks = self._retrieve_few(cleaned)
        else:
            chunks = self._retrieve_broad(cleaned)

        if not chunks:
            logger.warning("Retrieval returned no chunks | scope=%s", scope)
            return []

        logger.info(
            "Retrieval done | chunks=%d | sources=%d",
            len(chunks), len({c.source_file for c in chunks}),
        )
        return chunks

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 1 — SINGLE
    # User pointed at one specific document.
    # Build filter from filter_hints → filtered search.
    # If results < MIN_RESULTS_THRESHOLD → relax to filename tokens → retry.
    # If still thin → no filter at all.
    # ══════════════════════════════════════════════════════════════════
    def _retrieve_single(self, cleaned: CleanedQuery) -> list[RetrievedChunk]:
        hints       = cleaned.filter_hints
        query_set   = self._build_query_set(cleaned)
        filter_dict = self._build_filter(hints)

        logger.info("Single scope | filter=%s", filter_dict)

        # strict filtered search across all subqueries
        raw_pool, hit_counts = self._embed_and_search(query_set, filter_dict)

        # fallback 1: relax to filename tokens only
        if len(raw_pool) < config.MIN_RESULTS_THRESHOLD and filter_dict:
            relaxed = {}
            if hints.get("filename_tokens"):
                tokens = []
                for t in hints["filename_tokens"]:
                    tokens.extend(t.lower().split())
                relaxed = {"filename_or": tokens}

            logger.warning(
                "Strict filter returned %d results — relaxing | filter=%s",
                len(raw_pool), filter_dict
            )
            raw_pool, hit_counts = self._embed_and_search(query_set, relaxed)

            # fallback 2: no filter at all
            if len(raw_pool) < config.MIN_RESULTS_THRESHOLD:
                logger.warning(
                    "Relaxed filter still thin — falling back to no filter | filter=%s",
                    relaxed
                )
                raw_pool, hit_counts = self._embed_and_search(query_set, None)

        deduped  = self._deduplicate(raw_pool)
        reranked = self._rerank(cleaned.improved_query, deduped)
        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        return self._threshold_and_cap(boosted, config.PER_QUERY_TOP_K)

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 2 — FEW (comparison / multi-doc)
    # User named 2-4 specific documents or asked to compare.
    # Run one filtered search per arm, tag chunks with arm_label.
    # Merge all arms → rerank together.
    # ══════════════════════════════════════════════════════════════════
    def _retrieve_few(self, cleaned: CleanedQuery) -> list[RetrievedChunk]:
        arms      = cleaned.comparison_arms[:4]   # hard cap at 4
        query_set = self._build_query_set(cleaned)

        # fallback: no arms extracted → treat as single with filter_hints
        if not arms:
            logger.warning("Few scope but no comparison_arms — falling back to single")
            return self._retrieve_single(cleaned)

        all_chunks: list[RetrievedChunk] = []

        for arm in arms:
            filter_dict = self._build_filter_from_arm(arm)
            logger.info("Arm '%s' | filter=%s", arm.label, filter_dict)

            raw_pool, hit_counts = self._embed_and_search(query_set, filter_dict)

            # relax if arm returns too few
            if len(raw_pool) < config.MIN_RESULTS_THRESHOLD and filter_dict:
                logger.warning(
                    "Arm '%s' thin — relaxing | filter=%s", arm.label, filter_dict
                )
                relaxed = self._relax_filter(filter_dict)
                raw_pool, hit_counts = self._embed_and_search(query_set, relaxed)

            deduped  = self._deduplicate(raw_pool)
            reranked = self._rerank(cleaned.improved_query, deduped)
            boosted  = self._apply_boosts(reranked, cleaned, hit_counts)

            # tag every chunk with which arm it belongs to
            slots = max(1, config.PER_QUERY_TOP_K // len(arms))
            for chunk in boosted[:slots]:
                chunk.arm_label = arm.label
                all_chunks.append(chunk)

            logger.info(
                "Arm '%s' | chunks=%d", arm.label, len(boosted[:slots])
            )

        # sort within each arm by rerank score, preserve arm grouping
        all_chunks.sort(key=lambda c: (c.arm_label, -c.rerank_score))
        return all_chunks

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 3 — BROAD
    # No specific document named. Could be anywhere in corpus.
    # Unfiltered search → diversity cap by doc_id.
    # specificity=high → apply filename/keyword boosts after rerank.
    # ══════════════════════════════════════════════════════════════════
    def _retrieve_broad(self, cleaned: CleanedQuery) -> list[RetrievedChunk]:
        query_set = self._build_query_set(cleaned)

        # no filter — full corpus search
        raw_pool, hit_counts = self._embed_and_search(query_set, filter_dict=None)

        deduped  = self._deduplicate(raw_pool)
        reranked = self._rerank(cleaned.improved_query, deduped)
        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)

        # diversity cap: max MAX_CHUNKS_PER_DOC per doc, max MAX_DOCS_BROAD docs
        capped = self._diversity_cap(boosted)

        logger.info(
            "Broad scope | specificity=%s | after_cap=%d",
            cleaned.specificity, len(capped),
        )
        return capped

    # ══════════════════════════════════════════════════════════════════
    # CORE — embed + search across all subqueries
    # This is the same for all 3 strategies.
    # improved_query always leads. subqueries follow.
    # ══════════════════════════════════════════════════════════════════
    def _build_query_set(self, cleaned: CleanedQuery) -> list[str]:
        """
        improved_query is always first — it's the anchor.
        subqueries follow — they are only for finding chunks.
        LLM only ever sees improved_query, never subqueries.
        """
        queries = [cleaned.improved_query]
        for sq in cleaned.subqueries:
            if sq.query not in queries:
                queries.append(sq.query)
        return queries

    def _embed_and_search(
        self,
        queries:     list[str],
        filter_dict: dict | None,
    ) -> tuple[list[dict], dict[tuple, int]]:
        raw_pool:   list[dict]       = []
        hit_counts: dict[tuple, int] = defaultdict(int)

        for query in queries:
            dense, sparse = self._embed(query)
            if dense is None:
                continue

            results = self._search(dense, sparse, filter_dict)
            for r in results:
                key = (r.get("source_file", ""), r.get("chunk_index", 0))
                hit_counts[key] += 1
                raw_pool.append(r)

            logger.info(
                "Searched | query='%s...' | filter=%s | results=%d",
                query[:50], filter_dict, len(results),
            )

        return raw_pool, hit_counts

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        seen: dict[tuple, dict] = {}
        for r in results:
            key = (r.get("source_file", ""), r.get("chunk_index", 0))
            if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
                seen[key] = r
        return list(seen.values())

    # ══════════════════════════════════════════════════════════════════
    # RERANK — CrossEncoder scores every chunk against improved_query
    # improved_query is what user actually asked — reranker judges
    # relevance against that, not against subqueries.
    # ══════════════════════════════════════════════════════════════════
    def _rerank(
        self, improved_query: str, results: list[dict]
    ) -> list[RetrievedChunk]:
        if not results:
            return []

        # Use instruction configuration
        instruct_query = (
            f"Instruct: {config.RERANKER_INSTRUCTION}\n"
            f"Query: {improved_query}"
        )

        try:
            pairs  = [[instruct_query, r.get("text", "")] for r in results]
            scores = self.rerank_model.predict(pairs)

            scored = []
            for raw_score, result in zip(scores, results):
                sigmoid_score = 1 / (1 + math.exp(-float(raw_score)))
                scored.append((sigmoid_score, result))
                logger.info(
                    "Reranked | logit=%.4f | sigmoid=%.4f | file=%s | chunk=%d",
                    float(raw_score), sigmoid_score,
                    result.get("source_file", ""),
                    result.get("chunk_index", 0),
                )

            scored.sort(key=lambda x: x[0], reverse=True)

            relevant = [(s, r) for s, r in scored if s >= config.CONFIDENCE_THRESHOLD]
            logger.info(
                "Rerank done | top=%.4f | relevant=%d | total=%d",
                scored[0][0] if scored else 0.0,
                len(relevant), len(scored),
            )

            if relevant:
                logger.info("━━━ RELEVANT CHUNKS (%d) ━━━", len(relevant))
                for rank, (score, result) in enumerate(relevant, 1):
                    preview = result.get("text", "").replace("\n", " ").strip()[:120]
                    logger.info(
                        "  #%d | score=%.4f | chunk=%d | page=%s | '%s...'",
                        rank, score,
                        result.get("chunk_index", 0),
                        result.get("page_label", result.get("page_no", "?")),
                        preview,
                    )
                logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            else:
                logger.warning("No chunks passed threshold %.2f", config.CONFIDENCE_THRESHOLD)

            return [
                self._to_chunk(result, rerank_score=score)
                for score, result in scored
            ]

        except Exception as e:
            logger.error("Reranking failed | %s — falling back to Qdrant score", e)
            results.sort(key=lambda r: r.get("score", 0), reverse=True)
            return [
                self._to_chunk(r, rerank_score=r.get("score", 0.0))
                for r in results
            ]

    # ══════════════════════════════════════════════════════════════════
    # BOOSTS — small nudges after rerank, never override reranker
    # Reads from filter_hints (new model), not scope (old model)
    # ══════════════════════════════════════════════════════════════════
    def _apply_boosts(
        self,
        chunks:     list[RetrievedChunk],
        cleaned:    CleanedQuery,
        hit_counts: dict[tuple, int],
    ) -> list[RetrievedChunk]:
        hints        = cleaned.filter_hints
        year         = (hints.get("doc_year") or "").lower()
        section      = (hints.get("section") or "").lower()
        keywords     = [k.lower() for k in hints.get("keywords", [])]
        fn_tokens    = [t.lower() for t in hints.get("filename_tokens", [])]

        for chunk in chunks:
            boost = 0.0
            key   = (chunk.source_file, chunk.chunk_index)

            # table boost — tables are dense with facts
            if chunk.is_table:
                boost += config.TABLE_BOOST

            # year match in chunk
            if year and chunk.chunk_year:
                if year in chunk.chunk_year:
                    boost += config.YEAR_BOOST

            # section match
            if section and chunk.section:
                if section in chunk.section.lower():
                    boost += config.SECTION_BOOST

            # keyword match in chunk text
            if keywords:
                chunk_lower = chunk.text.lower()
                if any(k in chunk_lower for k in keywords):
                    boost += config.KEYWORD_BOOST
                else:
                    if cleaned.target_scope in ("single", "few"):
                        boost += config.ENTITY_PENALTY

            # filename token match in source_file
            if fn_tokens:
                source_lower = chunk.source_file.lower()
                if any(t in source_lower for t in fn_tokens):
                    boost += config.FILENAME_TOKEN_BOOST

            # multi-subquery hit bonus
            hit_count = hit_counts.get(key, 1)
            if hit_count > 1:
                boost += config.HIT_COUNT_BOOST * (hit_count - 1)

            chunk.rerank_score += boost

        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks

    # ── Diversity cap for broad scope ─────────────────────────────────
    def _diversity_cap(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Broad scope: max MAX_CHUNKS_PER_DOC per doc, max MAX_DOCS_BROAD docs.
        Preserves score ordering within each doc.
        """
        doc_buckets: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            key = chunk.doc_id if chunk.doc_id else chunk.source_file
            doc_buckets[key].append(chunk)

        result    = []
        doc_count = 0
        for doc_key, doc_chunks in doc_buckets.items():
            if doc_count >= config.MAX_DOCS_BROAD:
                break
            result.extend(doc_chunks[:config.MAX_CHUNKS_PER_DOC])
            doc_count += 1

        return result

    # ── Threshold + cap ────────────────────────────────────────────────
    def _threshold_and_cap(
        self, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        filtered = [c for c in chunks if c.rerank_score >= config.CONFIDENCE_THRESHOLD]
        if not filtered:
            logger.warning(
                "All chunks below threshold %.2f", config.CONFIDENCE_THRESHOLD
            )
            return []
        return filtered[:top_k]

    # ── Filter builders ────────────────────────────────────────────────
    def _build_filter(self, hints: dict) -> dict:
        """
        Build Qdrant filter_dict from filter_hints.
        Only year_or filter used as strict filter.
        filename_tokens and keywords moved to boosts only.
        """
        f = {}
        if hints.get("doc_year"):
            f["year_or"] = hints["doc_year"]  # cross-field OR: doc_year OR chunk_year
        return f

    def _build_filter_from_arm(self, arm: ComparisonArm) -> dict:
        """
        Build Qdrant filter_dict from a single comparison arm.
        Only year_or — filename_tokens handled by boosts.
        """
        f = {}
        if arm.year:
            f["year_or"] = arm.year  # cross-field OR: doc_year OR chunk_year
        return f

    def _relax_filter(self, filter_dict: dict) -> dict:
        """
        Used by _retrieve_few arm fallback.
        Year OR filter failed — drop everything, reranker takes over.
        """
        return {}

    # ── Embed + search helpers ─────────────────────────────────────────
    def _embed(self, query: str):
        try:
            dense, sparse = self.embedder.embed_query(query)
            return dense, sparse
        except Exception as e:
            logger.error("Query embedding failed | %s", e)
            return None, None

    def _search(
        self, dense, sparse, filter_dict: dict | None
    ) -> list[dict]:
        try:
            return self.db.search_hybrid(
                query_dense  = dense,
                query_sparse = sparse,
                filter_dict  = filter_dict,
                top_k        = config.TOP_K_SEARCH,
            )
        except Exception as e:
            logger.error("Qdrant search failed | %s", e)
            return []

    # ── Chunk formatter for LLM ────────────────────────────────────────
    def format_for_llm(self, chunks: list[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks for LLM context block.
        Used by assembler. LLM only sees improved_query + this block.
        """
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            if chunk.arm_label:
                label = chunk.arm_label
            elif chunk.is_weak_match:
                label = "[LOW CONFIDENCE]"
            else:
                label = chunk.section or "General"

            header = f"--- Context {i} {chunk.source_tag} | {label} ---"
            parts.append(f"{header}\n{chunk.text}\n")

        return "\n".join(parts)

    # ── Dict → RetrievedChunk ──────────────────────────────────────────
    def _to_chunk(self, result: dict, rerank_score: float) -> RetrievedChunk:
        source_file = result.get("source_file", "")
        page_label  = result.get("page_label", "")
        page_no     = result.get("page_no", 0)

        if page_label:
            source_tag = f"[{source_file}, Page {page_label}]"
        elif page_no > 0:
            source_tag = f"[{source_file}, Page {page_no}]"
        else:
            source_tag = f"[{source_file}]"

        return RetrievedChunk(
            text                  = result.get("text", ""),
            source_file           = source_file,
            page_no               = page_no,
            page_label            = page_label,
            chunk_index           = result.get("chunk_index", 0),
            section               = result.get("section", ""),
            is_table              = result.get("is_table", False),
            doc_year              = result.get("doc_year", ""),
            doc_id                = result.get("doc_id", ""),
            chunk_year            = result.get("chunk_year", []),
            token_count           = result.get("token_count", 0),
            qdrant_score          = result.get("score", 0.0),
            rerank_score          = rerank_score,
            is_weak_match         = rerank_score < config.CONFIDENCE_THRESHOLD,
            source_tag            = source_tag,
            arm_label             = "",   # set by _retrieve_few() after this
            summary               = result.get("summary", ""),
        )

def get_retriever(embedder: OllamaEmbedder) -> Retriever:
    return Retriever(embedder=embedder, db=embedder.db)