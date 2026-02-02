import json
import boto3
from deepeval.models.base_model import DeepEvalBaseLLM
from src.models.token_usage import TokenUsage


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
            model_name=self.model_id,
        )

        content = response_body.get("content", [])
        response_text = content[0].get("text", "") if content else ""

        self._current_turn_tokens = token_usage
        return response_text, token_usage

    @property
    def last_token_usage(self) -> TokenUsage | None:
        return self._current_turn_tokens


class BedrockModelWrapper(DeepEvalBaseLLM):
    def __init__(self, model: str, region_name: str):
        self.name = model
        self.region_name = region_name
        self.model = self.load_model(self.name, self.region_name)
        self.token_history: list[TokenUsage] = []

    def load_model(self, model_id: str, region_name: str) -> BedrockLLMClient:
        return BedrockLLMClient(model_id, region_name)

    def generate(self, prompt: str, template_name: str = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response_text, token_usage = self.model.invoke_model(messages, **kwargs)
        if template_name is not None:
            token_usage.turn_name = template_name
        self.token_history.append(token_usage)
        return response_text

    async def a_generate(self, prompt: str, template_name: str = None, **kwargs) -> str:
        return self.generate(prompt, template_name=template_name, **kwargs)

    def get_model_name(self) -> str:
        return self.name

    def clear_token_history(self):
        self.token_history = []
