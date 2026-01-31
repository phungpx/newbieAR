import json
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from src.models.doc_info import DocInfo, DocStatus, ChunkMetadata
from src.ingestion.config import CHUNKS_DIR, MODEL_ID, CHUNKER_CONFIG


class DoclingChunker:
    def __init__(
        self,
        output_dir: Path = CHUNKS_DIR,
        tokenizer_name: str = MODEL_ID,
        max_tokens: int = None,
    ):
        self.output_dir = Path(output_dir)
        self.tokenizer_name = tokenizer_name

        # Initialize tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)

        # Initialize chunker
        chunker_config = CHUNKER_CONFIG.copy()
        if max_tokens:
            chunker_config["max_tokens"] = max_tokens

        self.chunker = HybridChunker(tokenizer=self.tokenizer, **chunker_config)

        self.converter = DocumentConverter()

    def chunk_document(self, file_path: str) -> tuple[list[ChunkMetadata], DocInfo]:
        file_path = Path(file_path)

        doc_info = DocInfo(
            file_name=file_path.name,
            format=file_path.suffix,
            status=DocStatus.CHUNKING.value,
        )

        try:
            logger.info(f"Chunking {file_path.name}...")

            # Load document
            loaded_document = self.converter.convert(source=str(file_path)).document

            # Chunk document
            chunks = self._process_chunks(loaded_document)

            # Save chunks
            chunk_path = self._save_chunks(file_path, chunks)

            # Update doc info
            doc_info.chunk_path = str(chunk_path)
            doc_info.chunk_count = len(chunks)
            doc_info.status = DocStatus.SUCCESS.value

            logger.info(f"✓ Chunked {file_path.name} successfully")
            logger.info(f"  Chunks: {len(chunks)}")
            logger.info(f"  Output: {chunk_path}")

            return chunks, doc_info

        except Exception as e:
            logger.error(f"✗ Failed to chunk {file_path.name}: {e}")
            doc_info.status = DocStatus.FAILED.value
            doc_info.error = str(e)
            return [], doc_info

    def _process_chunks(self, document) -> list[ChunkMetadata]:
        chunks = []

        for i, chunk in enumerate(self.chunker.chunk(document)):
            text_tokens = self.tokenizer.count_tokens(chunk.text)
            contextualized_text = self.chunker.contextualize(chunk=chunk)
            contextualized_tokens = self.tokenizer.count_tokens(contextualized_text)

            chunk_metadata = ChunkMetadata(
                chunk_id=i,
                text=chunk.text,
                text_tokens=text_tokens,
                contextualized_text=contextualized_text,
                contextualized_tokens=contextualized_tokens,
                filename=chunk.meta.origin.filename,
                mimetype=chunk.meta.origin.mimetype,
            )
            chunks.append(chunk_metadata)

            logger.debug(
                f"  Chunk #{i + 1}: {text_tokens} tokens "
                f"({contextualized_tokens} contextualized)"
            )

        return chunks

    def _save_chunks(self, file_path: Path, chunks: list[ChunkMetadata]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"{file_path.stem}.json"
        chunk_data = [chunk.model_dump() for chunk in chunks]

        with output_file.open(mode="w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=4, ensure_ascii=False)

        return output_file


def chunk_document(
    file_path: str,
    output_dir: str = None,
    tokenizer_name: str = MODEL_ID,
    max_tokens: int = None,
) -> dict:
    chunker = DoclingChunker(
        output_dir=output_dir or CHUNKS_DIR,
        tokenizer_name=tokenizer_name,
        max_tokens=max_tokens,
    )

    chunks, doc_info = chunker.chunk_document(file_path)

    return {
        "chunks": [chunk.model_dump() for chunk in chunks],
        "doc_info": doc_info.model_dump(),
    }


if __name__ == "__main__":
    test_files = ["data/wikipedia/Albert_Einstein.pdf"]

    chunker = DoclingChunker()
    for file_path in test_files:
        chunks, doc_info = chunker.chunk_document(file_path)
        print(json.dumps(doc_info.model_dump(), indent=4))
        print(f"Generated {len(chunks)} chunks")
