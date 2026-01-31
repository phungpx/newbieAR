from pathlib import Path
from loguru import logger
from docling.document_converter import DocumentConverter


class DocLoader:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.converter = DocumentConverter()

    def convert(self, file_path: str) -> str:
        """Convert a document to markdown format.
        Args:
            file_path: Path to the input document
        Returns:
            markdown path
        """
        try:
            logger.info(f"Converting {file_path.name} to markdown...")

            # Convert document
            result = self.converter.convert(source=str(file_path))
            markdown = result.document.export_to_markdown()

            # Save markdown file
            markdown_path = self._save_markdown(file_path, markdown)

            logger.info(f"Converted {file_path.name} successfully")
            logger.info(f"Output: {markdown_path}")
            logger.info(f"Length: {len(markdown)} characters")

            return markdown_path

        except Exception as e:
            logger.error(f"✗ Failed to convert {file_path.name}: {e}")
            raise e

    def _save_markdown(self, file_path: str, markdown: str) -> str:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{Path(file_path).stem}.md"
        output_file.write_text(markdown, encoding="utf-8")
        return str(output_file)


def convert_document_to_markdown(file_path: str, output_dir: str = None) -> dict:
    converter = DocLoader(output_dir=output_dir)
    doc_info = converter.convert(file_path)
    return doc_info.model_dump()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    converter = DocLoader(output_dir=args.output_dir)
    saved_path = converter.convert(args.file_path)
    print(f"Saved to: {saved_path}")
