from sentence_transformers import SentenceTransformer
import openai
from tqdm import tqdm
from typing import Generator


def chunks(lst: list, size: int | None = None) -> list | Generator:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]


class SentenceTransformerEmbedding:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        model_dim: int = 1024,
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.model_dim = model_dim

    def embed(self, inputs: list[str]) -> list[list[float]]:
        embeddings = []
        for chunk in tqdm(chunks(inputs, self.batch_size)):
            embeddings.extend(self.model.encode(chunk).tolist())

        return embeddings


class OpenAIEmbedding:
    def __init__(self, base_url: str, api_key: str, model_id: str):
        self.model = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def embed(self, inputs: list[str]) -> list[list[float]]:
        embeddings = self.model.embeddings.create(
            model=self.model_id,
            input=inputs,
            encoding_format="float",
        )
        embeddings = [data.embedding for data in embeddings.data]
        return embeddings


if __name__ == "__main__":
    text = "Hello World!"
    # embed = SentenceTransformerEmbedding(model_name="thenlper/gte-base").embed([text])
    # print(len(embed[0]))

    embed = OpenAIEmbedding(
        base_url="http://127.0.0.1:1234/v1",
        api_key="empty",
        model_id="text-embedding-all-minilm-l6-v2-embedding",
    ).embed([text])
    print(len(embed[0]))
