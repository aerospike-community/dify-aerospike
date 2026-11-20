"""``BaseVector`` implementation over a shared Aerospike set.

Data model: every dataset's chunks live in one configured set. The primary key
is ``<collection_name>:<doc_id>``; each record carries ``page_content``,
``metadata`` (map), ``vector``, and the promoted ``dataset_id`` /
``document_id`` bins. Every operation is scoped to the bound ``dataset_id``.

Invariants: a write replaces the whole record (no stale bins); all validation
happens before any write; caller documents and embeddings are never mutated.
"""

from __future__ import annotations

import heapq
import math
import threading
from collections.abc import Iterator
from operator import itemgetter
from typing import Any

from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.models.document import Document
from dify_vdb_aerospike import filters as flt
from dify_vdb_aerospike.config import AerospikeConfig, load_dify_config
from dify_vdb_aerospike.connection import get_session, indexes_ready, mark_indexes_ready

VECTOR_TYPE = VectorType.AEROSPIKE
DEFAULT_TOP_K = 4
PUSHDOWN_TOP_K_CAP = 1000
WRITE_CHUNK = 200
DELETE_CHUNK = 500
INDEX_FIELDS = (flt.DATASET_BIN, flt.DOCUMENT_BIN)
SCORE_BIN = "score"


def _is_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json(v) for k, v in value.items())
    return False


def _validate_vector(vector: Any) -> int:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("embedding must be a non-empty list of numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vector):
        raise ValueError("embedding must contain only numbers")
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("embedding must contain only finite values")
    if not any(x != 0 for x in vector):
        raise ValueError("embedding must not be zero-magnitude")
    return len(vector)


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _validate_threshold(threshold: Any) -> float:
    if threshold is None:
        return 0.0
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("score_threshold must be a number")
    if not math.isfinite(threshold):
        raise ValueError("score_threshold must be finite")
    return float(threshold)


class AerospikeVector(BaseVector):
    def __init__(self, collection_name: str, config: AerospikeConfig | None = None):
        """Bind a collection and validated settings without connecting to the server."""
        super().__init__(collection_name)
        self._config = config or load_dify_config()

    # -- connection (lazy: constructing an adapter never touches the server) --

    @property
    def session(self):
        return get_session(self._config)

    @property
    def _dataset(self):
        from aerospike_sdk import DataSet

        return DataSet.of(self._config.namespace, self._config.set_)

    def _pk(self, doc_id: str) -> str:
        return f"{self._collection_name}:{doc_id}"

    def _strip(self, pk: str) -> str:
        return pk[len(self._collection_name) + 1 :]

    def _scoped_query(self, *exprs):
        self._ensure_indexes_once()
        query = self.session.query(self._dataset).filter(flt.dataset_filter(self._collection_name))
        for expr in exprs:
            query = query.where(expr)
        if self._config.rps:
            query = query.records_per_second(self._config.rps)
        return query

    def _scoped_rows(self, query) -> Iterator:
        """Failed rows must surface as errors — silently dropping them would
        make delete/lookup report success over a partial result set."""
        for row in query.execute():
            if not row.is_ok:
                row.or_raise()
            if row.record is not None:
                yield row

    def get_type(self) -> str:
        return VECTOR_TYPE

    # -- index setup --------------------------------------------------------

    _index_lock = threading.Lock()

    def _ensure_indexes_once(self) -> None:
        """Create the shared indexes at most once per live connection;
        every scoped query path goes through here so reads before the first
        create still see the index. Readiness lives on the (config, pid)
        connection entry, so a reconnect after ``close_all`` re-runs the
        bootstrap."""
        if not indexes_ready(self._config):
            with AerospikeVector._index_lock:
                if not indexes_ready(self._config):
                    session = self.session
                    self._ensure_indexes()
                    mark_indexes_ready(self._config, session)

    def _ensure_indexes(self) -> None:
        cfg = self._config
        existing = {
            idx.get("name")
            for idx in self.session.list_indexes()
            if idx.get("namespace") == cfg.namespace and idx.get("set") == cfg.set_
        }
        for field in (field for field in INDEX_FIELDS if f"{cfg.set_}_{field}_idx" not in existing):
            name = f"{cfg.set_}_{field}_idx"
            try:
                self.session.index(dataset=self._dataset).on_bin(field).named(
                    name
                ).string().create().wait_till_complete_blocking()
            except Exception as exc:  # noqa: BLE001
                if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
                    raise
                if not self._index_exists(name):
                    raise
                # concurrent setup or same-name index: the existing one must be ours
        for field in INDEX_FIELDS:
            self._check_index_type(f"{cfg.set_}_{field}_idx", field)

    def _index_exists(self, name: str) -> bool:
        return any(i.get("name") == name for i in self.session.list_indexes())

    def _check_index_type(self, name: str, bin_name: str) -> None:
        for idx in (idx for idx in self.session.list_indexes() if idx.get("name") == name):
            bad_bin = idx.get("bin") not in (None, bin_name)
            bad_type = str(idx.get("type", "string")).lower() not in ("string", "text")
            if bad_bin or bad_type:
                raise RuntimeError(f"secondary index {name!r} exists with a conflicting definition")

    # -- writes ---------------------------------------------------------------

    def _prepare(self, documents: list[Document], embeddings: list[list[float]]) -> list[tuple[str, dict]]:
        if len(documents) != len(embeddings):
            raise ValueError(f"documents ({len(documents)}) and embeddings ({len(embeddings)}) length mismatch")
        dims = {_validate_vector(e) for e in embeddings}
        if len(dims) > 1:
            raise ValueError("embeddings in one call must share one dimension")
        prepared: dict[str, dict] = {}
        for doc, embedding in zip(documents, embeddings, strict=True):
            metadata = dict(doc.metadata or {})
            doc_id = metadata.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("every document needs a non-empty string metadata['doc_id']")
            if not _is_json(metadata):
                raise TypeError("metadata values must be JSON-compatible")
            parent = metadata.get(flt.DOCUMENT_BIN)
            if parent is not None and not isinstance(parent, str):
                raise TypeError("metadata['document_id'] must be a string")
            bins: dict[str, Any] = {
                flt.CONTENT_BIN: doc.page_content,
                flt.METADATA_BIN: metadata,
                flt.VECTOR_BIN: list(embedding),
                flt.DATASET_BIN: self._collection_name,
            }
            if parent is not None:
                bins[flt.DOCUMENT_BIN] = parent
            prepared.pop(doc_id, None)  # repeated doc_id in one call: last one wins
            prepared[doc_id] = bins
        return list(prepared.items())

    def _write(self, items: list[tuple[str, dict]]) -> list[str]:
        """Replace whole records in chunks of ``WRITE_CHUNK`` and return the written ids.

        Precondition: ``items`` was produced by ``_prepare`` (validated, unique ids).
        Postcondition: every id is returned only after its chunk's batch write fully
        succeeded; a failed chunk raises ``AerospikeError`` and earlier chunks stay written.

        Known limitation: this method depends on non-public preview SDK internals, namely
        ``session._behavior``, ``session.client.underlying_client.batch_write_blocking`` and
        ``policy_mapper.to_batch_policy``. They match the SDK commit pinned in this
        package's ``pyproject.toml`` but are not public API, so an SDK bump can break
        writes at runtime without an import-time error. Migrate to the public SDK write
        API once Aerospike publishes a stable vector release (vector 1.0) and re-verify
        batch policy, ``send_key`` and replace semantics at that point.
        """
        from aerospike_async import BatchWritePolicy, ExpOperation, RecordExistsAction  # noqa: F401
        from aerospike_sdk import ResultCode, Vector
        from aerospike_sdk.exceptions import AerospikeError
        from aerospike_sdk.policy.behavior_settings import Mode, OpKind, OpShape
        from aerospike_sdk.policy.policy_mapper import to_batch_policy

        session = self.session
        behavior = session._behavior
        settings = behavior.get_settings(OpKind.WRITE_RETRYABLE, OpShape.BATCH, Mode.AP)
        batch_policy = to_batch_policy(settings)
        write_policy = BatchWritePolicy()
        write_policy.send_key = True
        write_policy.commit_level = settings.commit_level
        write_policy.durable_delete = settings.durable_delete
        write_policy.record_exists_action = RecordExistsAction.REPLACE  # whole-record replacement
        dataset = self._dataset
        written: list[str] = []
        for start in range(0, len(items), WRITE_CHUNK):
            chunk = items[start : start + WRITE_CHUNK]
            keys = [dataset.id(self._pk(doc_id)) for doc_id, _ in chunk]
            bins_list = [{**bins, flt.VECTOR_BIN: Vector(bins[flt.VECTOR_BIN])} for _, bins in chunk]
            results = session.client.underlying_client.batch_write_blocking(
                keys, bins_list, batch_policy=batch_policy, write_policy=write_policy
            )
            failures = [r for r in results if r.result_code != ResultCode.OK]
            if failures:
                first = failures[0]
                raise AerospikeError(
                    f"batch write failed for {len(failures)}/{len(results)} keys "
                    f"(first: {first.server_message or first.result_code})",
                    result_code=first.result_code,
                    in_doubt=first.in_doubt,
                )
            written.extend(doc_id for doc_id, _ in chunk)
        return written

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs) -> list[str]:
        items = self._prepare(texts, embeddings)
        self._ensure_indexes_once()
        return self._write(items)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs) -> list[str]:
        """Returns the ids actually written: repeated ids in one call collapse
        to one record (last wins), so they appear once."""
        return self._write(self._prepare(documents, embeddings))

    # -- key operations --------------------------------------------------------

    def text_exists(self, id: str) -> bool:
        from aerospike_sdk import ResultCode
        from aerospike_sdk.exceptions import AerospikeError

        try:
            self.session.get(self._dataset.id(self._pk(id)), bins=[flt.DATASET_BIN])
            return True
        except AerospikeError as exc:
            if exc.result_code == ResultCode.KEY_NOT_FOUND_ERROR:
                return False
            raise

    def _delete_keys(self, pks: list[str]) -> None:
        dataset = self._dataset
        for start in range(0, len(pks), DELETE_CHUNK):
            keys = [dataset.id(pk) for pk in pks[start : start + DELETE_CHUNK]]
            self.session.delete(*keys).execute()

    def delete_by_ids(self, ids: list[str]) -> None:
        self._delete_keys([self._pk(i) for i in ids])

    def _pks_where(self, *exprs) -> list[str]:
        query = self._scoped_query(*exprs).bins([flt.DATASET_BIN])
        return [str(row.record.key.value) for row in self._scoped_rows(query)]

    def get_ids_by_metadata_field(self, key: str, value: Any) -> list[str]:
        return [self._strip(pk) for pk in self._pks_where(flt.metadata_field_expr(key, value))]

    def delete_by_metadata_field(self, key: str, value: Any) -> None:
        self._delete_keys(self._pks_where(flt.metadata_field_expr(key, value)))

    def delete(self) -> None:
        self._delete_keys(self._pks_where())

    # -- search ------------------------------------------------------------------

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        return []

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        top_k = _validate_top_k(kwargs.get("top_k", DEFAULT_TOP_K))
        threshold = _validate_threshold(kwargs.get("score_threshold"))
        _validate_vector(query_vector)
        doc_ids = list(kwargs.get("document_ids_filter") or [])
        exprs = [flt.document_ids_expr(list(dict.fromkeys(doc_ids)))] if doc_ids else []
        scored = self._scored_candidates(query_vector, exprs, top_k)
        return [self._to_document(bins, score) for score, bins in scored if score >= threshold]

    def _projection(self, query_vector: list[float]):
        from aerospike_async import ExpOperation, Operation
        from aerospike_sdk import Exp, Vector

        score = Exp.cosine_similarity(Vector(list(query_vector)), Exp.vector_bin(flt.VECTOR_BIN))
        ops = [Operation.get_bin(b) for b in (flt.CONTENT_BIN, flt.METADATA_BIN)]
        ops.append(ExpOperation.read(SCORE_BIN, score))
        return ops

    def _scored_candidates(self, query_vector: list[float], exprs: list, top_k: int) -> list[tuple[float, dict]]:
        from aerospike_sdk import Order, OrderByType

        query = self._scoped_query(*exprs).with_op_projection(*self._projection(query_vector))
        if top_k <= PUSHDOWN_TOP_K_CAP:
            rows = self._scoped_rows(query.order_by(SCORE_BIN, OrderByType.DOUBLE, Order.DESC).top_k(top_k))
            return [(s, b) for s, b in map(self._row_score, rows)]
        cap = self._config.max_candidates
        candidates = []
        for scanned, row in enumerate(self._scoped_rows(query), start=1):
            if cap is not None and scanned > cap:
                raise RuntimeError(f"search exceeded AEROSPIKE_MAX_CANDIDATES ({cap})")
            candidates.append(self._row_score(row))
        return heapq.nlargest(top_k, candidates, key=itemgetter(0))

    @staticmethod
    def _row_score(row) -> tuple[float, dict]:
        """Extract the computed score from an OK row's bins.

        Precondition: ``row`` is an OK row with a record.
        Postcondition: returns (score, remaining bins) without mutating
        ``row.record.bins``. Failure behaviour: raises ``RuntimeError``
        naming the record key when the score bin is absent.
        """
        bins = dict(row.record.bins)
        if SCORE_BIN not in bins:
            raise RuntimeError(f"search result for record {row.record.key.value!r} has no {SCORE_BIN!r} bin")
        return float(bins.pop(SCORE_BIN)), bins

    @staticmethod
    def _to_document(bins: dict, score: float) -> Document:
        metadata = dict(bins.get(flt.METADATA_BIN) or {})
        metadata["score"] = score
        return Document(page_content=bins.get(flt.CONTENT_BIN) or "", metadata=metadata)
