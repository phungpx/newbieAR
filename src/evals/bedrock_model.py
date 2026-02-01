import json
import boto3
from loguru import logger
from dataclasses import dataclass
from deepeval.models.base_model import DeepEvalBaseLLM


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0  # Tokens written to cache
    cache_read_input_tokens: int = 0  # Tokens read from cache

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens

    @property
    def total_cached_tokens(self) -> int:
        return self.cache_creation_input_tokens + self.cache_read_input_tokens


class BedrockLLMClient:
    def __init__(self, model_id: str, region_name: str):
        self.model_id = model_id
        self.region_name = region_name
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self._current_turn_tokens: TokenUsage | None = None

    def invoke_model(self, messages: list[dict], **kwargs) -> tuple[str, TokenUsage]:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                **kwargs,
            }
        )
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        response_body = json.loads(response["body"].read())
        usage = response_body.get("usage", {})
        token_usage = TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        )

        content = response_body.get("content", [])
        response_text = content[0].get("text", "") if content else ""

        logger.debug(
            f"Bedrock API call - Input: {token_usage.input_tokens}, "
            f"Output: {token_usage.output_tokens}, "
            f"Cache Creation: {token_usage.cache_creation_input_tokens}, "
            f"Cache Read: {token_usage.cache_read_input_tokens}"
        )

        self._current_turn_tokens = token_usage
        return response_text, token_usage

    @property
    def last_token_usage(self) -> TokenUsage | None:
        return self._current_turn_tokens


class BedrockModelWrapper(DeepEvalBaseLLM):
    def __init__(self, model_id: str, region_name: str):
        self.model_id = model_id
        self.region_name = region_name
        self.llm_client = BedrockLLMClient(model_id, region_name)
        self.token_history: list[tuple[TokenUsage, str]] = []

    def generate(self, prompt: str, template_name: str = "user", **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response_text, tokens = self.llm_client.invoke_model(messages, **kwargs)
        self.token_history.append((tokens, template_name))
        return response_text

    async def a_generate(
        self, prompt: str, template_name: str = "user", **kwargs
    ) -> str:
        return self.generate(prompt, template_name=template_name, **kwargs)

    def get_model_name(self) -> str:
        return self.model_id

    def load_model(self):
        """No-op for Bedrock API."""

    def clear_token_history(self):
        self.token_history = []
