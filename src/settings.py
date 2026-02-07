import os
from abc import ABC
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectBaseSettings(BaseSettings, ABC):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OpenAILLMSettings(ProjectBaseSettings):
    llm_model: str
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384


class ConfidentSettings(ProjectBaseSettings):
    confident_api_key: str | None = None


class LangfuseSettings(ProjectBaseSettings):
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_base_url: str | None = None


class QdrantVectorStoreSettings(ProjectBaseSettings):
    qdrant_uri: str
    qdrant_api_key: str | None = None
    qdrant_collection_name: str


class OpenAIEmbeddingSettings(ProjectBaseSettings):
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int


class RerankerSettings(ProjectBaseSettings):
    reranker_base_url: str
    reranker_api_key: str
    reranker_model: str


class Neo4jGraphDBSettings(ProjectBaseSettings):
    graph_db_uri: str
    graph_db_username: str
    graph_db_password: str


class MinIOSettings(ProjectBaseSettings):
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_secure: bool = False


class CritiqueModelSettings(ProjectBaseSettings):
    critique_model_name: str
    critique_model_region_name: str


class APISettings(ProjectBaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]
    admin_api_key: str = "newbie_admin_dev_key_change_in_production"


class AuthSettings(ProjectBaseSettings):
    api_keys_storage: str = "memory"  # or "redis"
    rate_limit_enabled: bool = True
    rate_limit_standard: str = "100/minute"
    rate_limit_premium: str = "1000/minute"


class JobSettings(ProjectBaseSettings):
    job_storage: str = "memory"  # or "redis"
    job_timeout_seconds: int = 600
    max_file_size_mb: int = 50


class SessionSettings(ProjectBaseSettings):
    session_storage: str = "memory"  # or "redis"
    session_ttl_hours: int = 24
    max_history_messages: int = 20


class ProjectSettings(
    OpenAILLMSettings,
    LangfuseSettings,
    QdrantVectorStoreSettings,
    OpenAIEmbeddingSettings,
    ConfidentSettings,
    RerankerSettings,
    Neo4jGraphDBSettings,
    MinIOSettings,
    CritiqueModelSettings,
    APISettings,
    AuthSettings,
    JobSettings,
    SessionSettings,
):
    @property
    def openai_llm(self) -> OpenAILLMSettings:
        return OpenAILLMSettings(**self.model_dump())

    @property
    def qdrant_vector_store(self) -> QdrantVectorStoreSettings:
        return QdrantVectorStoreSettings(**self.model_dump())

    @property
    def langfuse(self) -> LangfuseSettings:
        return LangfuseSettings(**self.model_dump())

    @property
    def openai_embedding(self) -> OpenAIEmbeddingSettings:
        return OpenAIEmbeddingSettings(**self.model_dump())

    @property
    def confident(self) -> ConfidentSettings:
        return ConfidentSettings(**self.model_dump())

    @property
    def reranker(self) -> RerankerSettings:
        return RerankerSettings(**self.model_dump())

    @property
    def neo4j_graph_db(self) -> Neo4jGraphDBSettings:
        return Neo4jGraphDBSettings(**self.model_dump())

    @property
    def minio(self) -> MinIOSettings:
        return MinIOSettings(**self.model_dump())

    @property
    def critique_model(self) -> CritiqueModelSettings:
        return CritiqueModelSettings(**self.model_dump())

    @property
    def api(self) -> APISettings:
        return APISettings(**self.model_dump())

    @property
    def auth(self) -> AuthSettings:
        return AuthSettings(**self.model_dump())

    @property
    def jobs(self) -> JobSettings:
        return JobSettings(**self.model_dump())

    @property
    def sessions(self) -> SessionSettings:
        return SessionSettings(**self.model_dump())


settings = ProjectSettings()
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.langfuse_secret_key
os.environ["LANGFUSE_BASE_URL"] = settings.langfuse.langfuse_base_url

os.environ["CONFIDENT_METRIC_LOGGING_FLUSH"] = "1"
