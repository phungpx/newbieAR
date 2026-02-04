import openai


class OpenAIEmbedding:
    def __init__(self, base_url: str, api_key: str, model_id: str):
        self.model = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def embed_texts(self, inputs: list[str]) -> list[list[float]]:
        embeddings = self.model.embeddings.create(
            model=self.model_id,
            input=inputs,
            encoding_format="float",
        )
        embeddings = [data.embedding for data in embeddings.data]
        return embeddings


if __name__ == "__main__":
    text = "Hello World!"
    embed = OpenAIEmbedding(
        base_url="http://127.0.0.1:1234/v1",
        api_key="empty",
        model_id="text-embedding-all-minilm-l6-v2-embedding",
    ).embed_texts([text])
    print(len(embed[0]))
