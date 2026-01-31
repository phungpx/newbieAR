import json
from pathlib import Path
from loguru import logger
from docling.document_converter import DocumentConverter
from src.models.doc_info import DocInfo, DocStatus
from src.ingestion.config import MARKDOWN_DIR


class DoclingDocumentConverter:
    def __init__(self, output_dir: Path = MARKDOWN_DIR):
        self.output_dir = Path(output_dir)
        self.converter = DocumentConverter()

    def convert(self, file_path: str) -> DocInfo:
        """Convert a document to markdown format.
        Args:
            file_path: Path to the input document
        Returns:
            DocInfo object with conversion results
        """
        file_path = Path(file_path)

        doc_info = DocInfo(
            file_name=file_path.name,
            format=file_path.suffix,
            status=DocStatus.CONVERTING.value,
        )

        try:
            logger.info(f"Converting {file_path.name} to markdown...")

            # Convert document
            result = self.converter.convert(source=str(file_path))
            markdown = result.document.export_to_markdown()

            # Update doc info
            doc_info.markdown_length = len(markdown)
            doc_info.markdown_preview = markdown[:200].replace("\n", " ")

            # Save markdown file
            markdown_path = self._save_markdown(file_path, markdown)
            doc_info.markdown_path = str(markdown_path)
            doc_info.status = DocStatus.SUCCESS.value

            logger.info(f"✓ Converted {file_path.name} successfully")
            logger.info(f"  Output: {markdown_path}")
            logger.info(f"  Length: {len(markdown)} characters")

        except Exception as e:
            logger.error(f"✗ Failed to convert {file_path.name}: {e}")
            doc_info.status = DocStatus.FAILED.value
            doc_info.error = str(e)

        return doc_info

    def _save_markdown(self, file_path: Path, markdown: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.output_dir / f"{file_path.stem}.md"
        output_file.write_text(markdown, encoding="utf-8")

        return output_file


def convert_document_to_markdown(file_path: str, output_dir: str = None) -> dict:
    converter = DoclingDocumentConverter(output_dir=output_dir or MARKDOWN_DIR)
    doc_info = converter.convert(file_path)
    return doc_info.model_dump()


if __name__ == "__main__":
    test_files = [
        "data/wikipedia/Albert_Einstein.pdf",
        "data/wikipedia/Isaac_Newton.txt",
    ]

    converter = DoclingDocumentConverter()
    for file_path in test_files:
        result = converter.convert(file_path)
        print(json.dumps(result.model_dump(), indent=2))
