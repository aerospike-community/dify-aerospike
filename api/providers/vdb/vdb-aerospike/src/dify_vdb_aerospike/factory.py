"""Dify ``AbstractVectorFactory`` for the Aerospike provider.

``init_vector`` never commits, rolls back, or closes the caller-owned session;
it only reads through it, and never connects to Aerospike (adapters connect
lazily on first operation).
"""

from __future__ import annotations

import json
from typing import Any

from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from dify_vdb_aerospike.aerospike_vector import VECTOR_TYPE, AerospikeVector


def _index_struct_dict(dataset: Any) -> dict | None:
    raw = getattr(dataset, "index_struct", None)
    return json.loads(raw) if raw else None


class AerospikeVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset, attributes: list, embeddings, *, session=None) -> AerospikeVector:
        """Resolve a dataset's collection without connecting or taking ownership of its SQL session."""
        binding_id = getattr(dataset, "collection_binding_id", None)
        index_struct = _index_struct_dict(dataset)
        if binding_id:
            from sqlalchemy import select

            from extensions.ext_database import db
            from models.dataset import DatasetCollectionBinding

            stmt = select(DatasetCollectionBinding).where(DatasetCollectionBinding.id == binding_id)
            binding = (session if session is not None else db.session).scalars(stmt).one_or_none()
            if binding is None:
                raise ValueError("Dataset Collection Bindings does not exist!")
            collection_name = binding.collection_name
        elif index_struct:
            collection_name = index_struct["vector_store"]["class_prefix"]
        else:
            from models.dataset import Dataset

            collection_name = Dataset.gen_collection_name_by_id(dataset.id)

        if not index_struct:
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VECTOR_TYPE, collection_name))
        return AerospikeVector(collection_name)
