from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Generator


def chunks(lst: list, size: int | None = None) -> list | Generator:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]


class EmbeddingClient:
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


if __name__ == "__main__":
    text = "Hello World!"
    embed = EmbeddingClient(model_name="thenlper/gte-large").embed([text])
    print(len(embed[0]))
