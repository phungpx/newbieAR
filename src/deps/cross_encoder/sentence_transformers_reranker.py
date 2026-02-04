import asyncio
import sentence_transformers

from .client import CrossEncoder as CrossEncoderClient


# https://huggingface.co/BAAI/bge-reranker-v2-m3


class SentenceTransformersReranker(CrossEncoderClient):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = sentence_transformers.CrossEncoder(model_name)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        input_pairs = [(query, passage) for passage in passages]

        # Run the synchronous predict method in an executor
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self.model.predict, input_pairs)

        ranked_passages = sorted(
            [
                (passage, float(score))
                for passage, score in zip(passages, scores, strict=False)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked_passages


if __name__ == "__main__":
    import asyncio

    async def main():
        reranker = SentenceTransformersReranker(model_name="BAAI/bge-reranker-v2-m3")
        query = "How many people live in Berlin?"
        passages = [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin is well known for its museums.",
        ]
        ranked_passages = await reranker.rank(query, passages)
        print(ranked_passages)

    asyncio.run(main())

    # Results:
    # [
    #   (
    #       'Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
    #       0.998177170753479
    #   ),
    #   (
    #       'Berlin is well known for its museums.',
    #       0.0002764426462817937
    #   )
    # ]
