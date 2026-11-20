"""Validated provider settings shared by Dify and explicit test configurations.

Invariants: a loaded ``AerospikeConfig`` always has a valid port range, a
non-empty namespace/set, and paired credentials. Secrets never appear in
``repr``/``str`` or in error messages.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3000
DEFAULT_NAMESPACE = "test"
DEFAULT_SET = "dify_vectors"

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TlsConfig:
    ca_file: str | None = None
    tls_name: str | None = None
    client_cert_file: str | None = None
    client_key_file: str | None = None


@dataclass(frozen=True)
class AerospikeConfig:
    """Seeds is a tuple of ``(host, port, tls_name | None)``; it takes
    precedence over host/port when non-empty."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    seeds: tuple[tuple[str, int, str | None], ...] = ()
    namespace: str = DEFAULT_NAMESPACE
    set_: str = DEFAULT_SET
    user: str | None = None
    password: str | None = field(default=None, repr=False)
    tls: TlsConfig | None = None
    max_candidates: int | None = None
    rps: int | None = None

    @property
    def seed_hosts(self) -> tuple[tuple[str, int, str | None], ...]:
        return self.seeds or ((self.host, self.port, self.tls.tls_name if self.tls else None),)


def _port(text: str, name: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise ValueError(f"{name} must be an integer port") from None
    if not 0 < port < 65536:
        raise ValueError(f"{name} must be in 1..65535")
    return port


def _positive_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_seeds(raw: str, default_port: int) -> tuple[tuple[str, int, str | None], ...]:
    """Grammar: comma-separated ``host[:port[:tls_name]]`` entries."""
    seeds = []
    for entry in (e.strip() for e in raw.split(",") if e.strip()):
        parts = entry.split(":")
        if len(parts) > 3 or not parts[0]:
            raise ValueError("AEROSPIKE_SEEDS entries must be host[:port[:tls_name]]")
        port = _port(parts[1], "AEROSPIKE_SEEDS port") if len(parts) > 1 and parts[1] else default_port
        seeds.append((parts[0], port, parts[2] if len(parts) > 2 and parts[2] else None))
    return tuple(seeds)


def load_config(env: Mapping[str, str] | None = None) -> AerospikeConfig:
    """Read ``AEROSPIKE_*`` variables (default ``os.environ``).

    Precondition: values are well-formed. Postcondition: returns a validated
    config; raises ``ValueError`` (without echoing secret values) otherwise.
    """
    env = os.environ if env is None else env
    port = _port(env.get("AEROSPIKE_PORT") or str(DEFAULT_PORT), "AEROSPIKE_PORT")
    namespace = env.get("AEROSPIKE_NAMESPACE") or DEFAULT_NAMESPACE
    set_ = env.get("AEROSPIKE_SET") or DEFAULT_SET
    user, password = env.get("AEROSPIKE_USER") or None, env.get("AEROSPIKE_PASSWORD") or None
    if (user is None) != (password is None):
        raise ValueError("AEROSPIKE_USER and AEROSPIKE_PASSWORD must be set together")
    tls_fields = {
        "ca_file": env.get("AEROSPIKE_TLS_CA_FILE") or None,
        "tls_name": env.get("AEROSPIKE_TLS_NAME") or None,
        "client_cert_file": env.get("AEROSPIKE_TLS_CERT_FILE") or None,
        "client_key_file": env.get("AEROSPIKE_TLS_KEY_FILE") or None,
    }
    if bool(tls_fields["client_cert_file"]) != bool(tls_fields["client_key_file"]):
        raise ValueError("AEROSPIKE_TLS_CERT_FILE and AEROSPIKE_TLS_KEY_FILE must be set together")
    tls_enabled = env.get("AEROSPIKE_TLS_ENABLED", "").lower() in _TRUE or any(tls_fields.values())
    return AerospikeConfig(
        host=env.get("AEROSPIKE_HOST") or DEFAULT_HOST,
        port=port,
        seeds=_parse_seeds(env.get("AEROSPIKE_SEEDS", ""), port),
        namespace=namespace,
        set_=set_,
        user=user,
        password=password,
        tls=TlsConfig(**tls_fields) if tls_enabled else None,
        max_candidates=_positive_int(env, "AEROSPIKE_MAX_CANDIDATES"),
        rps=_positive_int(env, "AEROSPIKE_RPS"),
    )


def load_dify_config() -> AerospikeConfig:
    """Read Dify's resolved settings and validate them before constructing an adapter.

    Returns immutable provider settings without connecting; invalid pairings
    raise ValueError without exposing credentials.
    """
    from configs import dify_config

    values = dify_config.model_dump(
        include={name for name in type(dify_config).model_fields if name.startswith("AEROSPIKE_")}
    )
    return load_config({name: str(value) for name, value in values.items() if value is not None})
