from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider

from src.settings import settings


def get_openai_model(
    model_name: str = "gemini-2.5-flash",
    base_url: str = settings.llm_base_url,
    api_key: str = settings.llm_api_key,
) -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        settings=ModelSettings(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ),
    )


def get_google_vertex_model(
    model_name: str = "gemini-2.5-flash",
    project_id="vns-durian-traceability",
) -> GoogleModel:
    return GoogleModel(
        model_name=model_name,
        provider=GoogleProvider(project=project_id, vertexai=True),
        settings=ModelSettings(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ),
    )
