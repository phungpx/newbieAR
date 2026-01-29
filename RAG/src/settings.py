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


class LLMSettings(ProjectBaseSettings):
    llm_model: str = "gpt-4o-mini"
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


class QdrantSettings(ProjectBaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "wikipedia-collection"


class SentenceTransformerEmbeddingSettings(ProjectBaseSettings):
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 1


class OpenAIEmbeddingSettings(ProjectBaseSettings):
    embedding_base_url: str = "http://127.0.0.1:1234/v1"
    embedding_api_key: str = "empty"
    embedding_model_id: str = "text-embedding-all-minilm-l6-v2-embedding"
    embedding_dimensions: int = 384


class ProjectSettings(
    LLMSettings,
    LangfuseSettings,
    QdrantSettings,
    SentenceTransformerEmbeddingSettings,
    OpenAIEmbeddingSettings,
    ConfidentSettings,
):
    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(**self.model_dump())

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(**self.model_dump())

    @property
    def langfuse(self) -> LangfuseSettings:
        return LangfuseSettings(**self.model_dump())

    @property
    def sentence_transformer_embedding(self) -> SentenceTransformerEmbeddingSettings:
        return SentenceTransformerEmbeddingSettings(**self.model_dump())

    @property
    def openai_embedding(self) -> OpenAIEmbeddingSettings:
        return OpenAIEmbeddingSettings(**self.model_dump())

    @property
    def confident(self) -> ConfidentSettings:
        return ConfidentSettings(**self.model_dump())


settings = ProjectSettings()
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.langfuse_secret_key
os.environ["LANGFUSE_BASE_URL"] = settings.langfuse.langfuse_base_url

os.environ["OPENAI_API_KEY"] = "empty"
os.environ["LOCAL_EMBEDDING_API_KEY"] = "empty"
