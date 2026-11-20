import pytest
from dify_vdb_aerospike.config import load_config, load_dify_config


def test_defaults():
    cfg = load_config({})
    assert cfg.seed_hosts == (("localhost", 3000, None),)
    assert (cfg.namespace, cfg.set_) == ("test", "dify_vectors")
    assert cfg.tls is None


def test_explicit_settings():
    cfg = load_config(
        {"AEROSPIKE_HOST": "seed", "AEROSPIKE_PORT": "3100", "AEROSPIKE_NAMESPACE": "ns", "AEROSPIKE_SET": "chunks"}
    )
    assert (cfg.host, cfg.port, cfg.namespace, cfg.set_) == ("seed", 3100, "ns", "chunks")


def test_seed_precedence():
    cfg = load_config({"AEROSPIKE_PORT": "3999", "AEROSPIKE_SEEDS": "s1:3101,s2:3102:tlsname2, s3"})
    assert cfg.seed_hosts == (("s1", 3101, None), ("s2", 3102, "tlsname2"), ("s3", 3999, None))


@pytest.mark.parametrize(
    "settings",
    [
        {"AEROSPIKE_PORT": "bad"},
        {"AEROSPIKE_PORT": "0"},
        {"AEROSPIKE_PORT": "65536"},
        {"AEROSPIKE_SEEDS": "s1:bad"},
        {"AEROSPIKE_USER": "only-user"},
        {"AEROSPIKE_PASSWORD": "secret-marker"},
        {"AEROSPIKE_TLS_CERT_FILE": "/cert"},
        {"AEROSPIKE_TLS_KEY_FILE": "/key"},
        {"AEROSPIKE_MAX_CANDIDATES": "0"},
        {"AEROSPIKE_RPS": "-1"},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError, match="AEROSPIKE_") as exc:
        load_config(settings)
    assert "secret-marker" not in str(exc.value)


@pytest.mark.parametrize(
    "settings",
    [
        {"AEROSPIKE_TLS_ENABLED": "true"},
        {"AEROSPIKE_TLS_CA_FILE": "/ca"},
        {"AEROSPIKE_TLS_NAME": "cluster"},
        {"AEROSPIKE_TLS_CERT_FILE": "/cert", "AEROSPIKE_TLS_KEY_FILE": "/key"},
    ],
)
def test_tls(settings):
    assert load_config(settings).tls is not None


def test_credentials_are_redacted():
    cfg = load_config({"AEROSPIKE_USER": "user", "AEROSPIKE_PASSWORD": "secret-marker"})
    assert "secret-marker" not in repr(cfg)
    assert cfg.password == "secret-marker"


def test_dify_settings_bridge():
    from configs import dify_config

    cfg = load_dify_config()
    assert cfg.host == dify_config.AEROSPIKE_HOST
    assert cfg.port == dify_config.AEROSPIKE_PORT
    assert cfg.namespace == dify_config.AEROSPIKE_NAMESPACE
    assert cfg.set_ == dify_config.AEROSPIKE_SET
