"""
agent/ollama_reranker.py
Qwen3-Reranker via Ollama — drop-in for sentence_transformers.CrossEncoder.

Matches the httpx + async-batch pattern already used in ingestion/embedder.py
(see _encode_passages_async) rather than introducing a new HTTP style.

WHY THIS EXISTS
---------------
Qwen3-Reranker is not a regression-head cross-encoder — it's a causal LM
instruction-tuned to answer only "yes" or "no" to:
    "Judge whether the Document meets the requirements based on the
     Query and the Instruct provided."

The official transformers/vLLM usage never reads the generated word —
it reads the LOGITS for the "yes" / "no" tokens and does:
    score = softmax([logit_no, logit_yes])[1]

Calling Ollama normally only returns the generated text — a hard
yes/no with no granularity. This module asks Ollama for logprobs on
that single next token and reconstructs the same score Alibaba's own
code produces.

Requires an Ollama build with /api/generate logprobs support (added in
Ollama v0.12.11, Nov 2025). Verify with:
    curl <OLLAMA_BASE_URL>/api/generate -d '{"model":"<model>",
        "prompt":"hi","stream":false,"logprobs":true,"top_logprobs":5,
        "options":{"num_predict":1}}'
and confirm the response JSON has a non-empty "logprobs" array.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
from typing import Optional

import httpx

logger = logging.getLogger("agent.ollama_reranker")

_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

_PREFIX = f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_FAILED_SCORE = -10.0  # strongly "not relevant" sentinel for a failed call,
                        # not 0.0 — 0.0 would outrank a genuinely weak match
                        # once the sigmoid in retriever.py is applied


class OllamaReranker:
    """
    Usage (mirrors OllamaEmbedder's constructor shape):

        reranker = OllamaReranker(
            ollama_url = config.RERANKER_OLLAMA_URL,   # f"{OLLAMA_BASE_URL}/api/generate"
            model_name = config.OLLAMA_RERANKER_MODEL,
        )
        scores = reranker.predict(pairs)   # pairs = [[query, doc], ...]

    predict() returns a RAW LOGIT (yes_logprob - no_logprob), not a 0-1
    probability — on purpose, so retriever.py's existing
        sigmoid_score = 1 / (1 + math.exp(-raw_score))
    line keeps working unmodified.
    """

    def __init__(
        self,
        ollama_url:   str,
        model_name:   str,
        instruction:  str = "Retrieve passages that are relevant to the given query and contain useful information to answer it.",
        top_logprobs: int = 20,
        timeout:      float = 60.0,
        batch_size:   int = 8,   # concurrent in-flight requests, mirrors EMBED_BATCH_SIZE
    ):
        self.ollama_url   = ollama_url
        self.model_name   = model_name
        self.instruction  = instruction
        self.top_logprobs = top_logprobs
        self.timeout      = timeout
        self.batch_size   = batch_size

        logger.info(
            "OllamaReranker ready | model=%s | url=%s | batch=%d",
            model_name, ollama_url, batch_size,
        )

    def _format_prompt(self, query: str, doc: str) -> str:
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        return _PREFIX + body + _SUFFIX

    async def _score_one_async(
        self, query: str, doc: str, client: httpx.AsyncClient
    ) -> float:
        payload = {
            "model":  self.model_name,
            "prompt": self._format_prompt(query, doc),
            "raw":    True,
            "stream": False,
            "logprobs":     True,
            "top_logprobs": self.top_logprobs,
            "options": {"num_predict": 1, "temperature": 0.0},
        }

        for attempt in (1, 2):
            try:
                r = await client.post(self.ollama_url, json=payload)
                r.raise_for_status()
                data = r.json()
                return self._extract_score(data)
            except Exception as e:
                if attempt == 1:
                    logger.warning("Reranker call failed — retrying | %s", e)
                else:
                    logger.error("Reranker call failed after retry | %s", e)
                    return _FAILED_SCORE

    def _extract_score(self, data: dict) -> float:
        try:
            top = data["logprobs"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            logger.warning(
                "No logprobs in response — confirm Ollama version supports "
                "logprobs on /api/generate"
            )
            return _FAILED_SCORE

        yes_lp, no_lp = -1e4, -1e4
        for cand in top:
            tok = cand.get("token", "").strip().lower()
            if tok == "yes":
                yes_lp = max(yes_lp, cand["logprob"])
            elif tok == "no":
                no_lp = max(no_lp, cand["logprob"])

        if yes_lp == -1e4 and no_lp == -1e4:
            logger.warning(
                "Neither 'yes' nor 'no' in top_logprobs=%d — raise top_logprobs",
                self.top_logprobs,
            )

        return yes_lp - no_lp

    async def _score_batch_async(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[Optional[float]] = [None] * len(pairs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(pairs), self.batch_size):
                batch   = pairs[start: start + self.batch_size]
                tasks   = [self._score_one_async(q, d, client) for q, d in batch]
                results = await asyncio.gather(*tasks)
                scores[start: start + self.batch_size] = results

        return scores

    def predict(self, pairs: list[list[str]], **_ignored_kwargs) -> list[float]:
        pair_tuples = [(q, d) for q, d in pairs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self._score_batch_async(pair_tuples))
            return future.result()
