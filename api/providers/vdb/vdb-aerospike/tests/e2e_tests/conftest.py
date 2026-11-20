import os
import uuid

import pytest
from dify_vdb_aerospike import AerospikeVector, load_config
from dify_vdb_aerospike.connection import close_all


@pytest.fixture
def adapter_factory():
    """Require an explicit disposable endpoint; delete only this test's unique collections."""
    if os.environ.get("AEROSPIKE_TEST_DISPOSABLE") != "1":
        pytest.skip("Set AEROSPIKE_TEST_DISPOSABLE=1 and AEROSPIKE_TEST_* for a disposable preview server")
    config = load_config(
        {
            "AEROSPIKE_HOST": os.environ.get("AEROSPIKE_TEST_HOST", "localhost"),
            "AEROSPIKE_PORT": os.environ.get("AEROSPIKE_TEST_PORT", "3000"),
            "AEROSPIKE_NAMESPACE": os.environ.get("AEROSPIKE_TEST_NAMESPACE", "test"),
            "AEROSPIKE_SET": os.environ.get("AEROSPIKE_TEST_SET", "dify_vdb_test"),
        }
    )
    created = []

    def make():
        adapter = AerospikeVector(f"test_{uuid.uuid4().hex}", config=config)
        created.append(adapter)
        return adapter

    yield make
    errors = []
    for adapter in created:
        try:
            adapter.delete()
        except Exception as exc:
            errors.append(f"{adapter.collection_name}: {exc}")
    close_all()
    if errors:
        pytest.fail("Test-owned records may remain: " + "; ".join(errors))
