import asyncio
from loguru import logger
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType
from src.deps import GraphitiClient


async def ingest(
    file_paths: list[str],
    clear_existing_graphdb_data: bool = False,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 1024,
    output_dir: str | None = None,
):
    graphiti = await GraphitiClient().create_client(
        clear_existing_graphdb_data=clear_existing_graphdb_data,
        max_coroutines=1,
    )

    json_chunks: list[dict] = []
    for file_path in file_paths:
        logger.info(f"Loading and chunking document: {file_path}")
        json_chunks.extend(
            load_and_chunk_document(
                file_path,
                tokenizer_name=tokenizer_name,
                max_tokens=max_tokens,
                output_dir=output_dir,
            )
        )

    try:
        for json_chunk in json_chunks:
            filename = json_chunk["metadata"]["filename"]
            chunk_id = json_chunk["chunk_id"]
            text = json_chunk["text"]

            logger.info(f"Added episode: {filename} - Chunk #{chunk_id}")

            await graphiti.add_episode(
                name=f"{filename} - Chunk #{chunk_id}",
                episode_body=text,
                source=EpisodeType.text,
                source_description=filename,
                reference_time=datetime.now(timezone.utc),
            )
    finally:
        logger.info("Closing graphiti client")
        await graphiti.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, required=True)
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
    args = parser.parse_args()

    file_paths = [str(file_path) for file_path in Path(args.file_dir).glob("**/*.md")]
    asyncio.run(
        ingest(
            file_paths=file_paths,
            clear_existing_graphdb_data=args.clear_existing_graphdb_data,
            tokenizer_name=args.tokenizer_name,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )
    )
