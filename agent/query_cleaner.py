"""
agent/query_cleaner.py
"""

import json
import logging
import time
from typing import Any, Dict, List

import numpy as np
import ollama

from final_rag.agent.models import CleanedQuery, ComparisonArm, Subquery
import final_rag.config as config
from final_rag.prompts.cleaner_prompt import CLEANER_SYSTEM_PROMPT, build_cleaner_prompt

logger = logging.getLogger("agent.query_cleaner")

SUBQUERY_WEIGHT_THRESHOLD = config.SUBQUERY_WEIGHT_THRESHOLD


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class QueryCleaner:
    def __init__(
        self,
        model:       str = config.CLEANER_MODEL,
        embed_model: str = config.EMBED_MODEL,
        ollama_host: str = config.OLLAMA_BASE_URL,
    ):
        self.model       = model
        self.embed_model = embed_model
        self.client      = ollama.Client(host=ollama_host)

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed all texts in a single batch call."""
        response = self.client.embed(model=self.embed_model, input=texts)
        return response.embeddings

    def _score_subqueries(
        self, improved_query: str, subqueries: List[Dict]
    ) -> List[Dict]:
        """
        Score each subquery by cosine similarity against improved_query.
        Single batch embed — one round trip to Qwen.
        """
        if not subqueries:
            return []

        texts      = [improved_query] + [sq["query"] for sq in subqueries]
        embeddings = self._batch_embed(texts)
        anchor     = embeddings[0]

        scored = []
        for i, sq in enumerate(subqueries):
            weight = cosine_similarity(anchor, embeddings[i + 1])
            scored.append({"query": sq["query"], "weight": round(weight, 4)})

        return sorted(scored, key=lambda x: x["weight"], reverse=True)

    def _parse_raw(self, raw: str) -> Dict[str, Any]:
        """Strip markdown fences and parse JSON."""
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            elif raw.startswith("python"):
                raw = raw[6:].strip()
        return json.loads(raw)

    def clean_query(self, query: str) -> Dict[str, Any]:
        start       = time.perf_counter()
        user_prompt = build_cleaner_prompt(query)

        try:
            response = self.client.generate(
                model   = self.model,
                system  = CLEANER_SYSTEM_PROMPT,
                prompt  = user_prompt,
                options = {"temperature": 0.0},
                think   = False,    
            )

            parsed         = self._parse_raw(response.response.strip())
            improved_query = parsed.get("improved_query", query)

            # ── score + filter subqueries ──────────────────────────────
            raw_subqueries = parsed.get("subqueries", [])
            all_subqueries = self._score_subqueries(improved_query, raw_subqueries)
            filtered       = [
                sq for sq in all_subqueries
                if sq["weight"] >= SUBQUERY_WEIGHT_THRESHOLD
            ]

            result = {
                "original_query":   query,
                "improved_query":   improved_query,
                "detected_language": parsed.get("detected_language", "english"),
                "target_scope":     parsed.get("target_scope",     "broad"),
                "answer_structure": parsed.get("answer_structure", "direct"),
                "specificity":      parsed.get("specificity",      "low"),
                "filter_hints":     parsed.get("filter_hints",     {}),
                "comparison_arms":  parsed.get("comparison_arms",  []),
                "all_subqueries":   all_subqueries,
                "subqueries":       filtered,
                "processing_time_sec": round(time.perf_counter() - start, 3),
            }

            logger.info(
                "Cleaned | scope=%s | structure=%s | specificity=%s | subqueries=%d",
                result["target_scope"],
                result["answer_structure"],
                result["specificity"],
                len(filtered),
            )
            return result

        except Exception as e:
            logger.error("Query cleaning failed: %s", e)
            return {
                "original_query":      query,
                "improved_query":      query,
                "detected_language":   "english",
                "target_scope":        "broad",
                "answer_structure":    "direct",
                "specificity":         "low",
                "filter_hints":        {},
                "comparison_arms":     [],
                "all_subqueries":      [],
                "subqueries":          [],
                "processing_time_sec": round(time.perf_counter() - start, 3),
            }

    def clean(self, query: str, active_document: str = None) -> CleanedQuery:
        result = self.clean_query(query)

        subqueries = [
            Subquery(query=sq["query"], weight=sq["weight"])
            for sq in result.get("subqueries", [])
        ]
        all_subqueries = [
            Subquery(query=sq["query"], weight=sq["weight"])
            for sq in result.get("all_subqueries", [])
        ]

        # ── parse comparison arms ──────────────────────────────────────
        comparison_arms = [
            ComparisonArm(
                label           = arm.get("label", ""),
                year            = arm.get("year"),
                filename_tokens = arm.get("filename_tokens", []),
            )
            for arm in result.get("comparison_arms", [])[:4]  # hard cap at 4
        ]

        return CleanedQuery(
            original_query      = result["original_query"],
            improved_query      = result["improved_query"],
            detected_language   = result.get("detected_language", "english"),
            target_scope        = result["target_scope"],
            answer_structure    = result["answer_structure"],
            specificity         = result["specificity"],
            filter_hints        = result["filter_hints"],
            comparison_arms     = comparison_arms,
            active_document     = active_document,
            subqueries          = subqueries,
            all_subqueries      = all_subqueries,
            processing_time_sec = result["processing_time_sec"],
        )


def get_cleaner(model: str = config.CLEANER_MODEL) -> QueryCleaner:
    return QueryCleaner(model=model)