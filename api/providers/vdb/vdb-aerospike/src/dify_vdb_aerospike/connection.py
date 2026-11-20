"""Process-wide shared cluster and session.

Invariants: at most one live cluster per (config, pid); a forked worker never
reuses its parent's connection. Index readiness is part of the live
connection's state: it is tracked per (config, pid) entry and is discarded
with that entry by ``close_all``.
"""

from __future__ import annotations

import os
import threading
from itertools import starmap

from dify_vdb_aerospike.config import AerospikeConfig

_lock = threading.Lock()


class _Entry:
    __slots__ = ("cluster", "session", "indexes_ready")

    def __init__(self, cluster, session):
        self.cluster = cluster
        self.session = session
        self.indexes_ready = False


_shared: dict[tuple[AerospikeConfig, int], _Entry] = {}


def _connect(config: AerospikeConfig):
    from aerospike_sdk import Behavior, Host
    from aerospike_sdk.sync import ClusterDefinition

    definition = ClusterDefinition(hosts=list(starmap(Host, config.seed_hosts)))
    if config.tls is not None:
        builder = definition.with_tls_config_of()
        if config.tls.ca_file:
            builder.ca_file(config.tls.ca_file)
        if config.tls.tls_name:
            builder.tls_name(config.tls.tls_name)
        if config.tls.client_cert_file:
            builder.client_cert_file(config.tls.client_cert_file)
            builder.client_key_file(config.tls.client_key_file)
        builder.done()
    if config.user is not None:
        definition.with_native_credentials(config.user, config.password)
    cluster = definition.connect()
    # send_key so record keys (and thus chunk doc_ids) are recoverable from query results.
    behavior = Behavior.DEFAULT.derive_with_changes("dify-vdb-aerospike", send_key=True)
    return cluster, cluster.create_session(behavior)


def get_session(config: AerospikeConfig):
    """Return the shared session for ``config``, connecting lazily once."""
    key = (config, os.getpid())
    with _lock:
        if key not in _shared:
            cluster, session = _connect(config)
            _shared[key] = _Entry(cluster, session)
        return _shared[key].session


def indexes_ready(config: AerospikeConfig) -> bool:
    """Return whether indexes were verified for the live connection.

    Precondition: none; safe to call before any connection exists.
    Postcondition: returns True only when a live entry for (config, pid)
    exists and was marked ready via ``mark_indexes_ready``; returns False
    when there is no connection or it was never marked. No side effects.
    """
    with _lock:
        entry = _shared.get((config, os.getpid()))
        return entry is not None and entry.indexes_ready


def mark_indexes_ready(config: AerospikeConfig, session) -> None:
    """Mark the live connection for ``config`` as index-ready.

    Precondition: ``session`` is the object previously returned by
    ``get_session(config)`` and captured before index bootstrap ran.
    Postcondition: on success the live entry for (config, pid) is flagged
    ready. Failure behaviour: if there is no live entry, or the entry's
    session is not exactly ``session`` (a stale session from before
    ``close_all``), the call does nothing — readiness can never leak onto a
    different connection.
    """
    with _lock:
        entry = _shared.get((config, os.getpid()))
        if entry is not None and entry.session is session:
            entry.indexes_ready = True


def close_all() -> None:
    """Close every live cluster and empty the registry.

    Precondition: none; safe to call at any time.
    Postcondition: ``_shared`` is empty on both success and failure, and
    every previously registered cluster had ``close()`` attempted exactly
    once. On failure, all close() errors are raised together as an
    ``ExceptionGroup`` after every cluster has been attempted; a failing
    close never prevents the remaining clusters from being closed.
    """
    with _lock:
        entries = list(_shared.values())
        _shared.clear()
        errors = []
        for entry in entries:
            try:
                entry.cluster.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to close Aerospike clusters", errors)
