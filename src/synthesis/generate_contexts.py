import json
import random
from pathlib import Path
from loguru import logger
from uuid import uuid4
from deepeval.dataset.golden import Golden
from deepeval.models import DeepEvalBaseLLM

from .schema import ContextScore
from .prompts.context_evaluation import CONTEXT_EVALUATION
from src.deps import DocumentChunker, OpenAIEmbedding, QdrantVectorStore


def save_goldens_to_files(goldens: list[Golden], output_dir: str = "goldens"):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")

    for golden in goldens:
        file_dir = output_dir / Path(golden.source_file).stem
        if not file_dir.exists():
            file_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {file_dir}")

        file_path = file_dir / f"{uuid4()}.json"
        golden_data = golden.model_dump(by_alias=True, exclude_none=True)

        try:
            with file_path.open(mode="w", encoding="utf-8") as f:
                json.dump(golden_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")

    logger.info(f"\nSuccessfully saved {len(goldens)} files to '{output_dir}'.")


def generate_contexts(
    file_path: str,
    embedder: OpenAIEmbedding,
    vector_store: QdrantVectorStore,
    embedding_size: int,
    num_contexts: int = 5,
    context_size: int = 3,
) -> list[list[str]]:
    """Chunk a document with docling, embed all chunks, store in Qdrant, then
    build semantically coherent contexts by retrieving k-1 nearest neighbors
    for randomly selected seed chunks.

    Args:
        file_path: Path to the document (PDF, DOCX, etc.)
        embedder: OpenAIEmbedding instance for embedding chunk texts.
        vector_store: QdrantVectorStore instance for similarity search.
        embedding_size: Dimensionality of the embedding vectors.
        num_contexts: Number of contexts to build per document.
        context_size: Chunks per context (1 seed + context_size-1 neighbors).

    Returns:
        List of contexts, each a list of paragraph strings.
        Returns [] if the document produces no chunks.
    """
    chunker = DocumentChunker(strategy="hierarchical")
    chunks, _ = chunker.chunk_document(file_path)
    texts = [c.text for c in chunks]

    if not texts:
        logger.warning(f"No chunks extracted from {file_path}. Skipping.")
        return []

    vectors = embedder.embed_texts(texts)
    collection_name = f"synthesis_{Path(file_path).stem}"

    try:
        vector_store.create_collection(collection_name, embedding_size)
        vector_store.add_embeddings(
            collection_name,
            embeddings=vectors,
            payloads=[{"text": t, "chunk_idx": i} for i, t in enumerate(texts)],
            ids=list(range(len(texts))),
        )

        contexts = []
        for _ in range(min(num_contexts, len(texts))):
            seed_idx = random.randint(0, len(texts) - 1)
            seed_vec = vectors[seed_idx]
            results = vector_store.query(
                collection_name, seed_vec, top_k=context_size + 1
            )
            neighbors = [
                r for r in results.points if r.payload["chunk_idx"] != seed_idx
            ][: context_size - 1]
            context = [texts[seed_idx]] + [n.payload["text"] for n in neighbors]
            contexts.append(context)

        return contexts
    finally:
        vector_store.delete_collection(collection_name)


async def evaluate_chunk(model: DeepEvalBaseLLM, chunk: str) -> float:
    prompt = CONTEXT_EVALUATION.format(context=chunk)
    response, _ = await model.a_generate(prompt, schema=ContextScore)
    return (
        response.clarity + response.depth + response.structure + response.relevance
    ) / 4
