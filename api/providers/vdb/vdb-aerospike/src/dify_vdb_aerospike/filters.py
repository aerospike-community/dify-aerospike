"""Expression helpers. Every helper returns a server-side filter; values are
never interpolated into AEL strings.

Invariant: a query carries exactly one secondary-index ``Filter`` (the
``dataset_id`` one); all other constraints are expressions via ``where()``.
"""

from __future__ import annotations

from typing import Any

DATASET_BIN = "dataset_id"
DOCUMENT_BIN = "document_id"
METADATA_BIN = "metadata"
CONTENT_BIN = "page_content"
VECTOR_BIN = "vector"

PROMOTED_STRING_BINS = (DOCUMENT_BIN,)


def dataset_filter(collection_name: str):
    from aerospike_sdk import Filter

    return Filter.equal(DATASET_BIN, collection_name)


def document_ids_expr(document_ids: list[str]):
    """OR-chain over the promoted parent ``document_id`` bin."""
    from aerospike_sdk import Exp

    clauses = [Exp.eq(Exp.string_bin(DOCUMENT_BIN), Exp.string_val(d)) for d in document_ids]
    return clauses[0] if len(clauses) == 1 else Exp.or_(clauses)


def _exp_type(value: Any):
    from aerospike_sdk import ExpType

    if isinstance(value, bool):
        return ExpType.BOOL
    if isinstance(value, int):
        return ExpType.INT
    if isinstance(value, float):
        return ExpType.FLOAT
    if isinstance(value, str):
        return ExpType.STRING
    raise TypeError(f"unsupported metadata lookup value type: {type(value).__name__}")


def metadata_field_expr(key: str, value: Any):
    """Typed equality: the promoted ``document_id`` bin is authoritative for its
    key; all other keys match inside the metadata map. A value of a different
    type never matches (int 1 != \"1\", bool != int); absent keys never match."""
    from aerospike_sdk import Exp, MapReturnType

    expected_type = _exp_type(value)
    if key in PROMOTED_STRING_BINS:
        return Exp.eq(Exp.string_bin(key), Exp.string_val(value)) if isinstance(value, str) else Exp.val(False)
    field = Exp.map_get_by_key(MapReturnType.VALUE, expected_type, Exp.string_val(key), Exp.map_bin(METADATA_BIN), [])
    return Exp.eq(field, Exp.val(value))
