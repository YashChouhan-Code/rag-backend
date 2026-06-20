"""
qdrant_storage/store.py

Qdrant Database Manager — Single Collection Hybrid Architecture
Stack: Qdrant Local File Storage (No Docker)
Features: Payload Indexing, Weighted Hybrid Search (Dense 90% / Sparse 10%), Fast Document Deletion

"""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    Prefetch,
    PayloadSchemaType,
)

import final_rag.config as config

logger = logging.getLogger("qdrant_storage.store")

# ── Hybrid weights ─────────────────────────────────────────────────────────────
DENSE_WEIGHT  = 0.85
SPARSE_WEIGHT = 0.15


class QdrantManager:
    """
    Manages the local Qdrant database.
    Single-collection architecture with B-Tree Payload Indexing
    for fast metadata filtering across thousands of documents.
    """

    def __init__(self):
        self.storage_path    = config.QDRANT_STORAGE_PATH
        self.collection_name = config.QDRANT_COLLECTION_NAME
        self.storage_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Connecting to Qdrant at %s...", self.storage_path)
            self.client = QdrantClient(
                path=str(self.storage_path),
                timeout=30,
            )
            self.client.get_collections()
            logger.info("Qdrant connection established")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant: {e}") from e

    # ── Connection health check ────────────────────────────────────────
    def get_client(self) -> QdrantClient:
        """Get client with health check and auto-reconnect"""
        try:
            self.client.get_collections()
            return self.client
        except Exception as e:
            logger.warning("Client unhealthy, reconnecting: %s", e)
            self.client = QdrantClient(
                path=str(self.storage_path),
                timeout=30,
            )
            return self.client

    def _create_index_safe(self, field: str, schema: PayloadSchemaType) -> bool:
        """Create index with error handling"""
        try:
            self.client.create_payload_index(
                collection_name = self.collection_name,
                field_name      = field,
                field_schema    = schema,
            )
            logger.info("  ✓ Indexed: %s (%s)", field, schema)
            return True
        except Exception as e:
            logger.error("  ✗ Failed to index %s: %s", field, e)
            return False

    # ── Setup ──────────────────────────────────────────────────────────
    def setup_database(self) -> None:
        """
        Creates hybrid collection + payload indexes.
        Safe to call multiple times — skips if collection already exists.
        Validates and rebuilds missing indexes.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        indexes = [
            ("source_file",      PayloadSchemaType.KEYWORD),
            ("page_no",          PayloadSchemaType.INTEGER),
            ("is_table",         PayloadSchemaType.BOOL),
            ("doc_year",         PayloadSchemaType.KEYWORD),
            ("chunk_year",       PayloadSchemaType.KEYWORD),
            ("section",          PayloadSchemaType.KEYWORD),
            ("doc_id",           PayloadSchemaType.KEYWORD),
            ("chunk_id",         PayloadSchemaType.KEYWORD),
            ("filename_tokens",  PayloadSchemaType.KEYWORD),
            ("keywords",         PayloadSchemaType.KEYWORD),
        ]

        if self.collection_name in existing:
            logger.info("Collection '%s' already exists.", self.collection_name)
            for field, schema in indexes:
                self._create_index_safe(field, schema)
            return

        # ── Create new collection ──────────────────────────────────────
        logger.info("Creating collection: '%s'", self.collection_name)
        self.client.create_collection(
            collection_name       = self.collection_name,
            vectors_config        = {
                "dense": VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE)
            },
            sparse_vectors_config = {
                "sparse": SparseVectorParams(),
            },
        )

        logger.info("Building payload indexes...")
        failed_indexes = []
        for field, schema in indexes:
            if not self._create_index_safe(field, schema):
                failed_indexes.append(field)

        if failed_indexes:
            raise RuntimeError(
                f"Failed to create indexes on new collection: {failed_indexes}. "
                "Aborting to prevent degraded retrieval."
            )

        logger.info(
            "✓ Database setup complete. Collection '%s' ready with all indexes.",
            self.collection_name,
        )

    # ── Document management ────────────────────────────────────────────
    def delete_document(self, source_file: str) -> None:
        """
        Deletes all points for a source file. Called before re-ingestion.
        Uses wait=True to ensure completion before upsert.
        """
        try:
            logger.info("Deleting records for: %s", source_file)
            self.client.delete(
                collection_name = self.collection_name,
                points_selector = Filter(
                    must=[
                        FieldCondition(
                            key   = "source_file",
                            match = MatchValue(value=source_file),
                        )
                    ]
                ),
                wait=True,
            )
            logger.info("Delete completed for: %s", source_file)
        except Exception as e:
            logger.error("Failed to delete '%s': %s", source_file, e)
            raise

    # ── Filter builder ─────────────────────────────────────────────────
    def _build_filter(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        """
        Converts a plain dict into a Qdrant Filter.
        year_or      → should (OR) across doc_year + chunk_year fields
        filename_or  → should (OR) across filename_tokens field
        Single value → must (AND)
        List values  → should (OR) nested inside must
        """
        if not filter_dict:
            return None

        must_conditions   = []
        should_conditions = []

        for key, value in filter_dict.items():
            # cross-field OR for year: doc_year OR chunk_year
            if key == "year_or":
                should_conditions.extend([
                    FieldCondition(key="doc_year",   match=MatchValue(value=value)),
                    FieldCondition(key="chunk_year", match=MatchValue(value=value)),
                ])
            # OR across filename_tokens list — used in relaxed filter
            elif key == "filename_or":
                should_conditions.extend([
                    FieldCondition(key="filename_tokens", match=MatchValue(value=v))
                    for v in value
                ])
            elif isinstance(value, list):
                should_conditions.extend([
                    FieldCondition(key=key, match=MatchValue(value=v))
                    for v in value
                ])
            else:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if must_conditions and should_conditions:
            return Filter(
                must=must_conditions + [Filter(should=should_conditions)]
            )
        if should_conditions:
            return Filter(should=should_conditions)
        if must_conditions:
            return Filter(must=must_conditions)

        return None

    # ── Score normaliser ───────────────────────────────────────────────
    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        """
        Min-max normalise a list of scores to [0, 1].
        Returns all zeros if all scores are identical.
        """
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    # ── Hybrid search ──────────────────────────────────────────────────
    def search_hybrid(
        self,
        query_dense:  List[float],
        query_sparse: Dict[int, float],
        filter_dict:  Optional[Dict[str, Any]] = None,
        top_k:        int = config.PER_QUERY_TOP_K,
    ) -> List[Dict[str, Any]]:
        # ── Build sparse vector ────────────────────────────────────────
        sorted_pairs = sorted(query_sparse.items())
        sparse_vec = SparseVector(
            indices=[k for k, _ in sorted_pairs],
            values=[v for _, v in sorted_pairs],
        )

        # ── Build filter ───────────────────────────────────────────────
        search_filter = self._build_filter(filter_dict)

        try:
            client = self.get_client()

            # ── Fetch dense candidates ─────────────────────────────────
            dense_results = client.query_points(
                collection_name = self.collection_name,
                query           = query_dense,
                using           = "dense",
                limit           = top_k * 5,
                with_payload    = True,
                query_filter    = search_filter,
            )

            # ── Fetch sparse candidates ────────────────────────────────
            sparse_results = client.query_points(
                collection_name = self.collection_name,
                query           = sparse_vec,
                using           = "sparse",
                limit           = top_k * 2,
                with_payload    = True,
                query_filter    = search_filter,
            )

            # ── Build score maps keyed by point id ────────────────────
            dense_scores:  Dict[str, float] = {
                str(r.id): r.score for r in dense_results.points
            }
            sparse_scores: Dict[str, float] = {
                str(r.id): r.score for r in sparse_results.points
            }
            payload_map:   Dict[str, Any] = {
                str(r.id): r.payload for r in dense_results.points
            }
            # include any sparse-only hits in payload map
            for r in sparse_results.points:
                if str(r.id) not in payload_map:
                    payload_map[str(r.id)] = r.payload

            all_ids = list(payload_map.keys())
            if not all_ids:
                logger.info(
                    "✓ Hybrid search complete — returned 0/%d chunks (filter=%s)",
                    top_k, bool(filter_dict),
                )
                return []

            # ── Normalise each leg independently ───────────────────────
            raw_dense  = [dense_scores.get(i, 0.0)  for i in all_ids]
            raw_sparse = [sparse_scores.get(i, 0.0) for i in all_ids]

            norm_dense  = self._normalize(raw_dense)
            norm_sparse = self._normalize(raw_sparse)

            # ── Blend: 85% dense + 15% sparse ─────────────────────────
            blended = [
                DENSE_WEIGHT * d + SPARSE_WEIGHT * s
                for d, s in zip(norm_dense, norm_sparse)
            ]

            # ── Sort and take top_k ────────────────────────────────────
            ranked = sorted(
                zip(all_ids, blended),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            hits = []
            for point_id, score in ranked:
                payload = payload_map.get(point_id, {})
                hits.append({
                    "score":           score,
                    "text":            payload.get("text", ""),
                    "source_file":     payload.get("source_file", ""),
                    "page_no":         payload.get("page_no", 0),
                    "page_range":      payload.get("page_range", (0, 0)),
                    "section":         payload.get("section", ""),
                    "chunk_index":     payload.get("chunk_index", 0),
                    "is_table":        payload.get("is_table", False),
                    "doc_year":        payload.get("doc_year", ""),
                    "chunk_year":      payload.get("chunk_year", []),
                    "token_count":     payload.get("token_count", 0),
                    "doc_id":          payload.get("doc_id", ""),
                    "chunk_id":        payload.get("chunk_id", ""),
                    "filename_tokens": payload.get("filename_tokens", []),
                    "has_tables":      payload.get("has_tables", False),
                    "summary":         payload.get("summary", ""),
                    "keywords":        payload.get("keywords", []),
                    "page_label":      payload.get("page_label", ""),
                })

            logger.info(
                "✓ Hybrid search complete — returned %d/%d chunks (filter=%s) "
                "| dense=%.0f%% sparse=%.0f%%",
                len(hits), top_k, bool(filter_dict),
                DENSE_WEIGHT * 100, SPARSE_WEIGHT * 100,
            )
            return hits

        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            return []