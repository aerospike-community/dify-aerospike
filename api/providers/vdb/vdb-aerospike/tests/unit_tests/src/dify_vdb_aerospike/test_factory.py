import json
from importlib.metadata import entry_points

from dify_vdb_aerospike.factory import AerospikeVectorFactory
from sqlalchemy.orm import Session

from core.rag.datasource.vdb.vector_backend_registry import clear_vector_factory_cache, get_vector_factory_class
from core.rag.datasource.vdb.vector_type import VectorType
from models.dataset import Dataset


def test_installed_entry_point():
    (entry,) = entry_points(group="dify.vector_backends", name="aerospike")
    assert entry.load() is AerospikeVectorFactory
    clear_vector_factory_cache()
    assert get_vector_factory_class(VectorType.AEROSPIKE) is AerospikeVectorFactory


def test_factory_binds_existing_collection_without_sql_or_network():
    dataset = Dataset(
        id="dataset-id", index_struct=json.dumps({"type": "aerospike", "vector_store": {"class_prefix": "existing"}})
    )
    with Session() as session:
        adapter = AerospikeVectorFactory().init_vector(dataset, [], None, session=session)
        assert adapter.collection_name == "existing"
        assert not session.in_transaction()
        assert session.is_active


def test_factory_creates_index_struct_and_independent_scopes():
    factory = AerospikeVectorFactory()
    datasets = [Dataset(id=value) for value in ("dataset-a", "dataset-b")]
    adapters = [factory.init_vector(dataset, [], None) for dataset in datasets]
    assert adapters[0].collection_name != adapters[1].collection_name
    for dataset, adapter in zip(datasets, adapters, strict=True):
        assert json.loads(dataset.index_struct) == {
            "type": "aerospike",
            "vector_store": {"class_prefix": adapter.collection_name},
        }
