import dataclasses

import pytest
from dify_vdb_aerospike import connection
from dify_vdb_aerospike.config import AerospikeConfig


class _FakeCluster:
    def __init__(self, fail=False):
        self.session = object()
        self.closed = False
        self.fail = fail

    def close(self):
        if self.fail:
            raise RuntimeError("boom")
        self.closed = True


@pytest.fixture
def fake_connect(monkeypatch):
    clusters = []

    def connect(config):
        cluster = _FakeCluster()
        clusters.append(cluster)
        return cluster, cluster.session

    monkeypatch.setattr(connection, "_connect", connect)
    yield clusters
    connection.close_all()


@pytest.fixture
def config():
    return AerospikeConfig(namespace="test", set_="conn_test")


def test_indexes_ready_false_before_and_after_connect(fake_connect, config):
    assert not connection.indexes_ready(config)
    connection.get_session(config)
    assert not connection.indexes_ready(config)


def test_mark_then_ready(fake_connect, config):
    session = connection.get_session(config)
    connection.mark_indexes_ready(config, session)
    assert connection.indexes_ready(config)


def test_close_all_resets_readiness(fake_connect, config):
    session = connection.get_session(config)
    connection.mark_indexes_ready(config, session)
    assert connection.indexes_ready(config)
    connection.close_all()
    assert not connection.indexes_ready(config)
    connection.get_session(config)
    assert not connection.indexes_ready(config)


def test_stale_session_cannot_mark_new_connection(fake_connect, config):
    stale = connection.get_session(config)
    connection.close_all()
    fresh = connection.get_session(config)
    assert fresh is not stale
    connection.mark_indexes_ready(config, stale)
    assert not connection.indexes_ready(config)
    connection.mark_indexes_ready(config, fresh)
    assert connection.indexes_ready(config)


def test_close_all_clears_registry_when_a_close_fails(fake_connect, config, monkeypatch):
    other = dataclasses.replace(config, host="other")

    def connect_fail_second(cfg):
        cluster = _FakeCluster(fail=cfg == other)
        fake_connect.append(cluster)
        return cluster, cluster.session

    monkeypatch.setattr(connection, "_connect", connect_fail_second)
    session = connection.get_session(config)
    connection.mark_indexes_ready(config, session)
    connection.get_session(other)

    with pytest.raises(ExceptionGroup) as excinfo:
        connection.close_all()
    assert excinfo.group_contains(RuntimeError, match="boom")
    first, second = fake_connect[0], fake_connect[1]
    assert first.closed is True
    assert second.closed is False
    assert connection._shared == {}
    assert not connection.indexes_ready(config)


def test_close_all_attempts_every_cluster_when_first_close_fails(fake_connect, config, monkeypatch):
    other = dataclasses.replace(config, host="other")

    def connect_fail_first(cfg):
        cluster = _FakeCluster(fail=cfg == config)
        fake_connect.append(cluster)
        return cluster, cluster.session

    monkeypatch.setattr(connection, "_connect", connect_fail_first)
    connection.get_session(config)
    connection.get_session(other)

    with pytest.raises(ExceptionGroup):
        connection.close_all()
    first, second = fake_connect[0], fake_connect[1]
    assert first.closed is False
    assert second.closed is True


def test_get_session_reconnects_after_failed_close_all(fake_connect, config, monkeypatch):
    stale = connection.get_session(config)

    original = connection._connect

    def connect_fail(cfg):
        cluster = _FakeCluster(fail=True)
        fake_connect.append(cluster)
        return cluster, cluster.session

    monkeypatch.setattr(connection, "_connect", connect_fail)
    other = dataclasses.replace(config, host="other")
    connection.get_session(other)
    with pytest.raises(ExceptionGroup):
        connection.close_all()
    monkeypatch.setattr(connection, "_connect", original)

    fresh = connection.get_session(config)
    assert fresh is not stale
    assert not connection.indexes_ready(config)


def test_configs_tracked_independently(fake_connect, config):
    other = AerospikeConfig(namespace="test", set_="conn_other")
    session = connection.get_session(config)
    connection.mark_indexes_ready(config, session)
    assert connection.indexes_ready(config)
    assert not connection.indexes_ready(other)
    connection.get_session(other)
    assert not connection.indexes_ready(other)
