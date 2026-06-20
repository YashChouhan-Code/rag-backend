"""
agent/orchestrator.py

Orchestrator — final stage of the RAG pipeline.
Pipeline: QueryCleaner → Retriever → Assembler → Orchestrator → Response

"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Generator

import ollama

from final_rag.agent.models import OrchestratorResult, SourceInfo, AssembledResult, CleanedQuery
from final_rag.agent.query_cleaner import get_cleaner
from final_rag.agent.retriever import get_retriever
from final_rag.agent.assembler import get_assembler
from final_rag.ingestion.embedder import OllamaEmbedder
from final_rag.prompts.generator_prompt import get_generator_prompt
import final_rag.config as config

logger = logging.getLogger("agent.orchestrator")

# ── Constants ──────────────────────────────────────────────────────────
MODEL             = config.GENERATOR_MODEL
MAX_HISTORY_TURNS = 3
MAX_RETRIES       = 1
NUM_CTX           = config.NUM_CTX
MAX_TOKENS        = config.MAX_TOKENS

CLARIFICATION_MESSAGE = (
    "I couldn't find specific information about that in the uploaded documents. "
    "Could you clarify your question or provide more details? "
    "For example, which document or section are you referring to?"
)

FALLBACK_ERROR_MESSAGE = (
    "Something went wrong while generating the answer. Please try again."
)


# ── Orchestrator ───────────────────────────────────────────────────────
class Orchestrator:

    def __init__(self, embedder: OllamaEmbedder):
        self.embedder  = embedder
        self.cleaner   = get_cleaner(model=config.CLEANER_MODEL)
        self.retriever = get_retriever(embedder=embedder)
        self.assembler = get_assembler()
        self.client    = ollama.Client(host=config.OLLAMA_BASE_URL)
        logger.info("Orchestrator initialized | generator=%s", MODEL)

    # ── Main pipeline entry ────────────────────────────────────────────
    def run(
        self,
        query:           str,
        history:         list[dict] = None,
        active_document: str = None,
    ) -> Generator[str, None, None]:

        total_start = time.perf_counter()
        history     = history or []
        times: dict[str, float] = {}

        try:
            # ── Stage 1: Clean ─────────────────────────────────────────
            t       = time.perf_counter()
            cleaned = self.cleaner.clean(query, active_document=active_document)
            times["cleaner"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Cleaned | %.3fs | scope=%s structure=%s specificity=%s",
                times["cleaner"], cleaned.target_scope,
                cleaned.answer_structure, cleaned.specificity,
            )

            # ── Stage 2: Retrieve ──────────────────────────────────────
            t      = time.perf_counter()
            chunks = self.retriever.retrieve(cleaned)
            times["retriever"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Retrieved %d chunks | %.3fs",
                len(chunks), times["retriever"],
            )

            # ── Stage 3: Assemble ──────────────────────────────────────
            t         = time.perf_counter()
            assembled = self.assembler.assemble(cleaned, chunks)
            times["assembler"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Assembled | not_found=%s weak=%s "
                "tables=%s structure=%s sources=%d | %.3fs",
                assembled.not_found, assembled.has_weak_match,
                assembled.has_tables, assembled.answer_structure,
                assembled.sources_count, times["assembler"],
            )

            # no relevant chunks found — ask user to clarify
            if assembled.not_found:
                yield CLARIFICATION_MESSAGE
                return

            # ── Stage 4: Generate ──────────────────────────────────────
            history_str = self._format_history(history)
            t = time.perf_counter()

            yield from self._stream_generate(
                original_query    = query,
                improved_query    = cleaned.improved_query,
                detected_language = cleaned.detected_language,
                assembled         = assembled,
                history_str       = history_str,
                stage_start       = t,
            )
            times["generator"] = round(time.perf_counter() - t, 3)

            # ── Metadata suffix — sources for frontend citation ────────
            if assembled.sources:
                yield self._build_metadata_payload(assembled.sources)

            total = round(time.perf_counter() - total_start, 3)
            logger.info(
                "[Orchestrator] Done | total=%.3fs | times=%s",
                total, times,
            )

        except Exception as e:
            logger.error(
                "[Orchestrator] Pipeline failed: %s\n%s",
                e, traceback.format_exc(),
            )
            yield FALLBACK_ERROR_MESSAGE

    # ── Stream generator output ────────────────────────────────────────
    def _stream_generate(
        self,
        original_query:    str,
        improved_query:    str,
        detected_language: str,
        assembled:         AssembledResult,
        history_str:       str,
        stage_start:       float,
    ) -> Generator[str, None, None]:
        template = get_generator_prompt(assembled.answer_structure)
        prompt   = template.format(
            history_str       = history_str,
            original_query    = original_query,
            detected_language = detected_language,
            context_block     = assembled.context_block,
        )

        logger.info(
            "[Orchestrator] Starting stream | model=%s think=False num_ctx=%d max_tokens=%d",
            MODEL, NUM_CTX, MAX_TOKENS,
        )

        try:
            stream = self.client.chat(
                model      = MODEL,
                messages   = [{"role": "user", "content": prompt}],
                # keep_alive = 0,
                stream     = True,
                think      = False,
                options    = {
                    "temperature": config.GENERATOR_TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "num_ctx":     NUM_CTX,
                },
            )

            token_count      = 0
            think_chunks     = 0
            first_token_time = None

            for chunk in stream:
                if not hasattr(chunk, "message"):
                    logger.warning("[Orchestrator] Unexpected chunk format: %s", chunk)
                    continue

                msg = chunk.message

                # ── Check if thinking is leaking through ───────────────
                if hasattr(msg, "thinking") and msg.thinking:
                    think_chunks += 1
                    logger.warning(
                        "[Orchestrator] ⚠️  THINKING chunk detected! "
                        "think=False is NOT working | chunk_no=%d | preview=%s",
                        think_chunks, str(msg.thinking)[:120],
                    )

                token = msg.content
                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        logger.info(
                            "[Orchestrator] 🟢 First token received | "
                            "time_to_first_token=%.3fs | "
                            "thinking_was_silent=%s",
                            first_token_time - stage_start,
                            (first_token_time - stage_start) > 30,  # >30s gap = silent thinking
                        )
                    token_count += 1
                    yield token

            # ── Final stream summary ───────────────────────────────────
            logger.info(
                "[Orchestrator] Stream complete | "
                "total_tokens=%d | think_chunks=%d | think_off_working=%s",
                token_count,
                think_chunks,
                "✅ YES" if think_chunks == 0 else "❌ NO — thinking is leaking",
            )

        except Exception as e:
            logger.error("[Orchestrator] Stream failed: %s", e)
            yield FALLBACK_ERROR_MESSAGE

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _build_metadata_payload(sources: list[SourceInfo]) -> str:
        try:
            return f"__METADATA__:{json.dumps([s.model_dump() for s in sources])}"
        except Exception as e:
            logger.error("[Orchestrator] Metadata serialization failed: %s", e)
            return ""

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "No previous context."
        recent = history[-MAX_HISTORY_TURNS:]
        return "\n".join(
            f"User: {t.get('question', '')}\nAssistant: {t.get('answer', '')}"
            for t in recent
        )


def get_orchestrator(embedder: OllamaEmbedder) -> Orchestrator:
    return Orchestrator(embedder=embedder)