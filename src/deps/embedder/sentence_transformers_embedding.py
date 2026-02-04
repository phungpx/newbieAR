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
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.model_dim = model_dim

    def embed_texts(self, inputs: list[str]) -> list[list[float]]:
        embeddings = []
        for chunk in tqdm(chunks(inputs, self.batch_size)):
            embeddings.extend(self.model.encode(chunk).tolist())

        return embeddings
