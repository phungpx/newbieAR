import openai
from loguru import logger


class OpenAIEmbedding:
    def __init__(self, base_url: str, api_key: str, model_id: str, timeout: int = 10):
        self.model = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.timeout = timeout

    def embed_texts(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = self.model.embeddings.create(
                model=self.model_id,
                input=inputs,
                encoding_format="float",
                timeout=self.timeout,
            )
            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {e} ({self.timeout} seconds timeout)")
            return []


if __name__ == "__main__":
    text = "Hello World!"
    embed = OpenAIEmbedding(
        base_url="http://127.0.0.1:1234/v1",
        api_key="empty",
        model_id="text-embedding-snowflake-arctic-embed-l-v2.0",
    ).embed_texts([text])
    if embed:
        print(len(embed[0]))
    else:
        print("Error embedding texts")
