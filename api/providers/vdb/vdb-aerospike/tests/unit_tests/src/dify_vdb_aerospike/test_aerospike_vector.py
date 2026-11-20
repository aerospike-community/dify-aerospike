import copy
import types

import pytest
from dify_vdb_aerospike.aerospike_vector import AerospikeVector, _validate_threshold, _validate_top_k, _validate_vector
from dify_vdb_aerospike.config import load_config

from core.rag.models.document import Document


@pytest.fixture
def adapter():
    return AerospikeVector("test-offline", config=load_config({}))


def test_type_and_unsupported_fulltext(adapter):
    assert adapter.get_type() == "aerospike"
    assert adapter.search_by_full_text("anything") == []


def test_prepare_preserves_pairing_and_inputs(adapter):
    docs = [
        Document(page_content="A", metadata={"doc_id": "a"}),
        Document(page_content="B", metadata={"doc_id": "b", "document_id": "parent"}),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    before = copy.deepcopy((docs, vectors))
    items = adapter._prepare(docs, vectors)
    assert [(key, bins["page_content"], bins["vector"]) for key, bins in items] == [
        ("a", "A", [1.0, 0.0]),
        ("b", "B", [0.0, 1.0]),
    ]
    assert all(bins["dataset_id"] == "test-offline" for _, bins in items)
    assert "document_id" not in items[0][1]
    assert items[1][1]["document_id"] == "parent"
    assert (docs, vectors) == before


def test_duplicate_ids_last_wins(adapter):
    docs = [Document(page_content=value, metadata={"doc_id": "dup"}) for value in ("first", "last")]
    assert [(key, bins["page_content"]) for key, bins in adapter._prepare(docs, [[1.0], [2.0]])] == [("dup", "last")]


@pytest.mark.parametrize("doc_id", [None, "", 123, ["x"]])
def test_invalid_doc_ids(adapter, doc_id):
    with pytest.raises(ValueError, match="embedding|doc_id|documents|top_k|score_threshold"):
        adapter.add_texts([Document(page_content="text", metadata={"doc_id": doc_id})], [[1.0]])


def test_count_and_dimension_mismatch(adapter):
    docs = [Document(page_content="text", metadata={"doc_id": str(i)}) for i in range(2)]
    with pytest.raises(ValueError, match="embedding|doc_id|documents|top_k|score_threshold"):
        adapter.add_texts(docs, [[1.0]])
    with pytest.raises(ValueError, match="embedding|doc_id|documents|top_k|score_threshold"):
        adapter.add_texts(docs, [[1.0], [1.0, 2.0]])


@pytest.mark.parametrize("vector", [[], [0.0, 0.0], [float("nan")], [float("inf")], [True], ["1"]])
def test_invalid_vectors(vector):
    with pytest.raises(ValueError, match="embedding|doc_id|documents|top_k|score_threshold"):
        _validate_vector(vector)


@pytest.mark.parametrize("top_k", [0, -1, 2.5, "4", True])
def test_invalid_topk(top_k):
    with pytest.raises(ValueError, match="embedding|doc_id|documents|top_k|score_threshold"):
        _validate_top_k(top_k)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_invalid_threshold_nonfinite(threshold):
    with pytest.raises(ValueError, match="embedding|doc_id|documents|top_k|score_threshold"):
        _validate_threshold(threshold)


@pytest.mark.parametrize("threshold", [True, "high", object()])
def test_invalid_threshold_type(threshold):
    with pytest.raises(TypeError):
        _validate_threshold(threshold)


def _fake_row(bins, key="pk1"):
    record = types.SimpleNamespace(key=types.SimpleNamespace(value=key), bins=bins)
    return types.SimpleNamespace(is_ok=True, record=record)


def test_row_score_missing_score_bin_raises_runtime_error():
    row = _fake_row({"page_content": "x"}, key="pk1")
    with pytest.raises(RuntimeError, match="pk1.*score"):
        AerospikeVector._row_score(row)


def test_row_score_returns_score_and_remaining_bins_without_mutating():
    bins = {"score": 0.5, "page_content": "x", "metadata": {"doc_id": "d"}}
    score, remaining = AerospikeVector._row_score(_fake_row(bins))
    assert score == 0.5
    assert remaining == {"page_content": "x", "metadata": {"doc_id": "d"}}
    assert bins == {"score": 0.5, "page_content": "x", "metadata": {"doc_id": "d"}}


def test_search_by_vector_surfaces_missing_score_as_runtime_error(monkeypatch, adapter):
    monkeypatch.setattr(AerospikeVector, "_ensure_indexes_once", lambda self: None)

    class _FakeQuery:
        def with_op_projection(self, *ops):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def top_k(self, top_k):
            return self

        def execute(self):
            return iter([_fake_row({"page_content": "x"})])

    monkeypatch.setattr(AerospikeVector, "_scoped_query", lambda self, *exprs: _FakeQuery())
    with pytest.raises(RuntimeError, match="pk1.*score"):
        adapter.search_by_vector([1.0, 0.0])


def test_score_conversion_does_not_mutate_metadata():
    bins = {"page_content": "content", "metadata": {"doc_id": "id", "score": "stored"}}
    hit = AerospikeVector._to_document(bins, -0.5)
    assert hit.page_content == "content"
    assert hit.metadata == {"doc_id": "id", "score": -0.5}
    assert bins["metadata"]["score"] == "stored"


def test_ensure_indexes_reruns_after_close_all(monkeypatch, adapter):
    from dify_vdb_aerospike import connection

    calls = []

    class _FakeCluster:
        def __init__(self):
            self.session = object()

        def close(self):
            pass

    def connect(config):
        cluster = _FakeCluster()
        return cluster, cluster.session

    monkeypatch.setattr(connection, "_connect", connect)
    monkeypatch.setattr(AerospikeVector, "_ensure_indexes", lambda self: calls.append(1))
    try:
        adapter._ensure_indexes_once()
        adapter._ensure_indexes_once()
        assert len(calls) == 1
        connection.close_all()
        adapter._ensure_indexes_once()
        assert len(calls) == 2
    finally:
        connection.close_all()
