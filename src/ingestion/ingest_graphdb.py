import asyncio
from pathlib import Path
from loguru import logger
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType
from src.deps import GraphitiClient, DocumentChunker
from src.models import ChunkStrategy


class GraphitiIngestion:
    def __init__(
        self,
        clear_existing_graphdb_data: bool = False,
        max_coroutines: int = 1,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 1024,
        output_dir: str | None = None,
        chunk_strategy: str = ChunkStrategy.HIERARCHICAL.value,
    ):
        self.clear_existing_graphdb_data = clear_existing_graphdb_data
        self.max_coroutines = max_coroutines
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.output_dir = output_dir

        self.doc_chunker = DocumentChunker(
            tokenizer_name=self.tokenizer_name,
            max_tokens=self.max_tokens,
            output_dir=self.output_dir,
            strategy=chunk_strategy,
        )

        self.graphiti_client = GraphitiClient()
        self.graphiti = None

    async def initialize_graphiti_client(self):
        if self.graphiti is None:
            self.graphiti = await self.graphiti_client.create_client(
                clear_existing_graphdb_data=self.clear_existing_graphdb_data,
                max_coroutines=self.max_coroutines,
            )

    async def ingest_file(self, file_path: str, original_filename: str | None = None) -> dict:
        await self.initialize_graphiti_client()

        logger.info(f"Loading and chunking document: {file_path}")
        chunks, _ = self.doc_chunker.chunk_document(file_path)

        for chunk in chunks:
            filename = Path(chunk.filename).stem
            chunk_id = chunk.chunk_id

            group_id = f"file-{filename}-chunk-{chunk_id}"

            logger.info(f"Added episode: {group_id}")

            await self.graphiti.add_episode(
                name=group_id,
                episode_body=chunk.text,
                source=EpisodeType.text,
                source_description=filename,
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
            )

        return {
            "filename": original_filename or Path(file_path).name,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text_tokens": c.text_tokens,
                    "text_preview": c.text[:120],
                }
                for c in chunks
            ],
        }

    async def ingest_files(self, file_paths: list[str]):
        for file_path in file_paths:
            await self.ingest_file(file_path)

    async def close(self):
        await self.graphiti_client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument(
        "--clear_existing_graphdb_data",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        default=ChunkStrategy.HIERARCHICAL.value,
    )
    args = parser.parse_args()

    ingestion = GraphitiIngestion(
        clear_existing_graphdb_data=args.clear_existing_graphdb_data,
        tokenizer_name=args.tokenizer_name,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        chunk_strategy=args.chunk_strategy,
    )

    file_paths = [args.file_path]
    asyncio.run(ingestion.ingest_files(file_paths))
