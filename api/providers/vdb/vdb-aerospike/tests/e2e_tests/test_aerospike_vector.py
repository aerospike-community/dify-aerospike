from concurrent.futures import ThreadPoolExecutor

import pytest

from core.rag.models.document import Document


def document(doc_id, content="text", **metadata):
    return Document(page_content=content, metadata={"doc_id": doc_id, **metadata})


def test_ingestion_replacement_and_lifecycle(adapter_factory):
    adapter = adapter_factory()
    assert not adapter.text_exists("a")
    assert adapter.create([document("a", document_id="old", annotation_id="obsolete")], [[1.0, 0.0]]) == ["a"]
    assert adapter.text_exists("a")
    assert adapter.add_texts([document("a", "replacement", document_id="new")], [[0.0, 1.0]]) == ["a"]
    assert adapter.get_ids_by_metadata_field("document_id", "old") == []
    assert adapter.get_ids_by_metadata_field("annotation_id", "obsolete") == []
    assert adapter.get_ids_by_metadata_field("document_id", "new") == ["a"]
    assert adapter.search_by_vector([0.0, 1.0])[0].page_content == "replacement"
    adapter.delete_by_ids(["a", "missing"])
    adapter.delete_by_ids(["a"])
    adapter.delete_by_ids([])
    assert not adapter.text_exists("a")
    adapter.create([document("b", document_id="parent")], [[1.0]])
    adapter.delete_by_metadata_field("document_id", "parent")
    assert not adapter.text_exists("b")
    adapter.delete()
    adapter.delete()


def test_replacement_removes_omitted_promoted_bin(adapter_factory):
    adapter = adapter_factory()
    adapter.create([document("id", document_id="parent")], [[1.0]])
    adapter.add_texts([document("id")], [[1.0]])
    assert adapter.get_ids_by_metadata_field("document_id", "parent") == []


def test_typed_metadata(adapter_factory):
    adapter = adapter_factory()
    adapter.create([document("a", num=1), document("b", flag=True)], [[1.0], [1.0]])
    assert adapter.get_ids_by_metadata_field("num", "1") == []
    assert adapter.get_ids_by_metadata_field("num", 1) == ["a"]
    assert adapter.get_ids_by_metadata_field("flag", 1) == []
    assert adapter.get_ids_by_metadata_field("missing", "value") == []


def test_colliding_datasets_remain_isolated(adapter_factory):
    a, b = adapter_factory(), adapter_factory()
    for adapter in (a, b):
        adapter.create([document("same", "original", document_id="parent", annotation_id="shared")], [[1.0, 0.0]])
    a.add_texts([document("same", "changed", document_id="changed")], [[0.0, 1.0]])
    assert b.search_by_vector([1.0, 0.0])[0].page_content == "original"
    assert a.search_by_vector([1.0, 0.0], document_ids_filter=["parent"]) == []
    a.delete_by_metadata_field("document_id", "changed")
    assert not a.text_exists("same")
    assert b.get_ids_by_metadata_field("annotation_id", "shared") == ["same"]
    a.add_texts([document("same")], [[1.0, 0.0]])
    a.delete_by_ids(["same"])
    assert b.text_exists("same")
    a.delete()
    assert b.text_exists("same")


def test_exact_score_oracle_thresholds_and_ordering(adapter_factory):
    adapter = adapter_factory()
    ids = ["positive", "fraction", "zero", "negative"]
    adapter.create([document(value) for value in ids], [[1.0, 0.0], [3.0, 4.0], [0.0, 1.0], [-1.0, 0.0]])
    hits = adapter.search_by_vector([1.0, 0.0], top_k=50, score_threshold=-1.0)
    assert [hit.metadata["doc_id"] for hit in hits] == ids
    assert [hit.metadata["score"] for hit in hits] == pytest.approx([1.0, 0.6, 0.0, -1.0], rel=1e-6, abs=1e-9)
    assert len(adapter.search_by_vector([1.0, 0.0], score_threshold=1.0)) == 1
    assert len(adapter.search_by_vector([1.0, 0.0])) == 3
    assert len(adapter.search_by_vector([1.0, 0.0], score_threshold=None)) == 3
    for restriction in (None, []):
        assert len(adapter.search_by_vector([1.0, 0.0], document_ids_filter=restriction)) == 3


def test_filter_before_topk_and_metadata_preservation(adapter_factory):
    adapter = adapter_factory()
    adapter.create(
        [document(str(i), document_id="wrong") for i in range(5)]
        + [document("eligible", document_id="right", score="stored")],
        [[1.0, 0.0]] * 5 + [[0.6, 0.8]],
    )
    hits = adapter.search_by_vector([1.0, 0.0], top_k=1, document_ids_filter=["right", "right", "unknown"])
    assert [hit.metadata["doc_id"] for hit in hits] == ["eligible"]
    assert hits[0].metadata["score"] == pytest.approx(0.6)
    assert adapter.get_ids_by_metadata_field("score", "stored") == ["eligible"]
    assert len(adapter.search_by_vector([1.0, 0.0])) == 4


def test_multibatch_delete_and_recreate(adapter_factory):
    adapter = adapter_factory()
    docs = [document(str(i)) for i in range(501)]
    adapter.create(docs, [[1.0]] * len(docs))
    adapter.delete()
    assert all(not adapter.text_exists(str(i)) for i in range(501))
    adapter.create([document("new")], [[1.0]])
    assert adapter.text_exists("new")


def test_pushdown_boundary(adapter_factory):
    adapter = adapter_factory()
    groups = [("hi", 400, [1.0, 0.0]), ("mid", 600, [0.9, 0.43588989435406733]), ("lo", 200, [0.8, 0.6])]
    docs = [document(f"{prefix}{i}", document_id="band") for prefix, count, _ in groups for i in range(count)]
    vectors = [vector for _, count, vector in groups for _ in range(count)]
    adapter.create(docs, vectors)
    for k in (999, 1000, 1001):
        hits = adapter.search_by_vector([1.0, 0.0], top_k=k)
        assert len(hits) == k
        assert sum(hit.metadata["doc_id"].startswith("hi") for hit in hits) == 400
        assert sum(hit.metadata["doc_id"].startswith("lo") for hit in hits) == max(0, k - 1000)
        scores = [hit.metadata["score"] for hit in hits]
        assert scores == sorted(scores, reverse=True)
    assert (
        len(adapter.search_by_vector([1.0, 0.0], top_k=1001, score_threshold=0.85, document_ids_filter=["band"]))
        == 1000
    )


def test_shared_connection_and_validation_before_writes(adapter_factory):
    a, b = adapter_factory(), adapter_factory()
    assert a.session is b.session
    docs = [document("first"), document("second")]
    with pytest.raises(ValueError, match="zero-magnitude"):
        a.add_texts(docs, [[1.0], [0.0]])
    assert not a.text_exists("first")
    assert not a.text_exists("second")
    assert a.add_texts(docs, [[1.0], [2.0]]) == ["first", "second"]


def test_concurrent_upserts_are_coherent(adapter_factory):
    adapter = adapter_factory()
    adapter.create([], [])
    payloads = [f"payload-{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda value: adapter.add_texts([document("same", value, submitted=value)], [[1.0]]), payloads)
        )
    assert results == [["same"]] * 8
    (hit,) = adapter.search_by_vector([1.0])
    assert hit.page_content in payloads
    assert hit.metadata["submitted"] == hit.page_content
