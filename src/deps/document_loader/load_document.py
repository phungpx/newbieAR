from pathlib import Path
from loguru import logger
from docling.document_converter import DocumentConverter

from src.settings import settings
from src.deps.document_loader.ocr_processor import DocumentAIOCRProcessor


class DocumentLoader:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.converter = DocumentConverter()
        self.ocr_processor = self._build_ocr_processor()

    def _build_ocr_processor(self) -> DocumentAIOCRProcessor | None:
        doc_ai = settings.google_doc_ai
        if doc_ai.google_doc_ai_project_id and doc_ai.google_doc_ai_processor_id:
            try:
                return DocumentAIOCRProcessor()
            except Exception as e:
                logger.warning(f"Document AI OCR unavailable: {e}")
        return None

    def convert(self, file_path: str) -> str:
        """Convert a document to markdown format.
        Args:
            file_path: Path to the input document
        Returns:
            markdown path
        """
        try:
            if self.ocr_processor and self.ocr_processor.is_scanned_pdf(file_path):
                logger.info(
                    f"Scanned PDF detected – running Document AI OCR on {file_path}"
                )
                ocr_text = self.ocr_processor.ocr(file_path)
                markdown_path = self._save_markdown(file_path, ocr_text)
                logger.info(f"OCR complete for {file_path}")
                logger.info(f"Output: {markdown_path}")
                logger.info(f"Length: {len(ocr_text)} characters")
                return markdown_path

            logger.info(f"Converting {file_path} to markdown...")

            result = self.converter.convert(source=file_path)
            markdown = result.document.export_to_markdown()

            markdown_path = self._save_markdown(file_path, markdown)

            logger.info(f"Converted {file_path} successfully")
            logger.info(f"Output: {markdown_path}")
            logger.info(f"Length: {len(markdown)} characters")

            return markdown_path

        except Exception as e:
            logger.error(f"✗ Failed to convert {file_path}: {e}")
            raise e

    def _save_markdown(self, file_path: str, markdown: str) -> str:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{Path(file_path).stem}.md"
        output_file.write_text(markdown, encoding="utf-8")
        return str(output_file)


def convert_document_to_markdown(file_path: str, output_dir: str = None) -> dict:
    converter = DocumentLoader(output_dir=output_dir)
    doc_info = converter.convert(file_path)
    return doc_info.model_dump()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    converter = DocumentLoader(output_dir=args.output_dir)
    saved_path = converter.convert(args.file_path)
    print(f"Saved to: {saved_path}")
