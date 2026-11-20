from pydantic import Field
from pydantic_settings import BaseSettings


class AerospikeConfig(BaseSettings):
    """Optional CE-compatible Aerospike preview settings; credentials and TLS are not required."""

    AEROSPIKE_HOST: str = "localhost"
    AEROSPIKE_PORT: int = Field(default=3000, ge=1, le=65535)
    AEROSPIKE_SEEDS: str | None = None
    AEROSPIKE_NAMESPACE: str = "test"
    AEROSPIKE_SET: str = "dify_vectors"
    AEROSPIKE_USER: str | None = None
    AEROSPIKE_PASSWORD: str | None = Field(default=None, repr=False)
    AEROSPIKE_TLS_ENABLED: bool = False
    AEROSPIKE_TLS_CA_FILE: str | None = None
    AEROSPIKE_TLS_NAME: str | None = None
    AEROSPIKE_TLS_CERT_FILE: str | None = None
    AEROSPIKE_TLS_KEY_FILE: str | None = None
    AEROSPIKE_MAX_CANDIDATES: int | None = Field(default=None, gt=0)
    AEROSPIKE_RPS: int | None = Field(default=None, gt=0)
