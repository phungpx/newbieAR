import json
import time
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownParams
from src.models import ChunkInfo


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNKED_TOKENS = 1024
VALID_STRATEGIES = {"hybrid", "hierarchical"}


class ImgPlaceholderSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider for hierarchical chunking with image placeholders."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


class DocChunker:
    def __init__(
        self,
        strategy: str = "hybrid",
        tokenizer_name: str = MODEL_ID,
        max_tokens: int = MAX_CHUNKED_TOKENS,
        merge_peers: bool = True,
        always_emit_headings: bool = False,
        merge_list_items: bool = True,
        output_dir: str = None,
    ):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.output_dir = output_dir
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
        )
        if strategy == "hybrid":
            self.chunker = HybridChunker(
                tokenizer=self.tokenizer,
                max_tokens=max_tokens,
                merge_peers=merge_peers,
                always_emit_headings=always_emit_headings,
            )
        elif strategy == "hierarchical":
            self.serializer_provider = ImgPlaceholderSerializerProvider()
            self.chunker = HierarchicalChunker(
                serializer_provider=self.serializer_provider,
                merge_list_items=merge_list_items,
            )
        self.loader = DocumentConverter()

    def chunk_document(self, file_path: str) -> tuple[list[ChunkInfo], str]:
        try:
            logger.info(f"Loading document from {Path(file_path).name}...")
            t = time.time()
            document = self.loader.convert(source=file_path).document
            logger.info(f"Document loaded in {time.time() - t:.2f} seconds")

            logger.info("Chunking document...")
            t = time.time()
            chunk_iter = self.chunker.chunk(document)

            chunks = []
            for i, chunk in enumerate(chunk_iter):
                text_tokens = self.tokenizer.count_tokens(chunk.text)
                contextualized_text = self.chunker.contextualize(chunk=chunk)
                contextualized_tokens = self.tokenizer.count_tokens(contextualized_text)
                chunk_info = ChunkInfo(
                    chunk_id=i,
                    text=chunk.text,
                    text_tokens=text_tokens,
                    contextualized_text=contextualized_text,
                    contextualized_tokens=contextualized_tokens,
                    filename=chunk.meta.origin.filename,
                    mimetype=chunk.meta.origin.mimetype,
                )
                chunks.append(chunk_info)

                logger.debug(f"  Chunk #{i + 1}: {text_tokens} tokens ")

            logger.info(
                f"Document chunked into {len(chunks)} chunks in {time.time() - t:.2f} seconds"
            )
            output_path = self._save_chunks(file_path, chunks)
            logger.info(f"Chunks saved to {output_path}")
            return chunks, output_path

        except Exception as e:
            logger.error(f"✗ Failed to chunk {file_path.name}: {e}")
            raise e

    def _save_chunks(self, file_path: str, chunks: list[ChunkInfo]) -> str:
        if self.output_dir:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{Path(file_path).stem}.json"
            chunk_data = [chunk.model_dump() for chunk in chunks]

            with output_file.open(mode="w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=4, ensure_ascii=False)

            return str(output_file)
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    chunker = DocChunker(output_dir=args.output_dir)
    chunks, output_path = chunker.chunk_document(args.file_path)
    print(f"{len(chunks)} chunks saved to {output_path}")
