from openai import OpenAI


class OpenAILLMClient:
    def __init__(
        self, base_url: str, api_keys: str | list[str], model_id: str = "gemma-3-12b-it"
    ):
        if not api_keys or len(api_keys) == 0:
            raise ValueError("api_keys is required")

        if isinstance(api_keys, str):
            api_keys = [api_keys]

        self.clients = [
            OpenAI(base_url=base_url, api_key=api_key) for api_key in api_keys
        ]
        self.current_client_index = 0
        self.model_id = model_id

    @property
    def client(self) -> OpenAI:
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0,
        top_p: float = 0.1,
        max_tokens: int = 5000,
    ):
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content


if __name__ == "__main__":
    from src.settings import settings

    llm_client = OpenAILLMClient(
        base_url=settings.llm_base_url,
        api_keys=settings.llm_api_key,
        model_id=settings.llm_model,
    )
    completion = llm_client.chat_completion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    print(completion)
