import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from loguru import logger
from pypdf import PdfReader

from src.settings import settings

ONLINE_PROCESSING_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


class DocumentAIOCRProcessor:
    """Processes documents via Google Document AI Layout Parser for
    structured, layout-aware text extraction.

    Uses LayoutConfig with ChunkingConfig to produce chunked output that
    preserves document hierarchy (headings, tables, figures).
    """

    MIME_TYPES = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "webp": "image/webp",
    }

    DEFAULT_CHUNK_SIZE = 500
    CHUNK_SEPARATOR = "\n\n"

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        processor_id: str | None = None,
        chunk_size: int | None = None,
    ):
        doc_ai = settings.google_doc_ai
        self.project_id = project_id or doc_ai.google_doc_ai_project_id
        self.location = location or doc_ai.google_doc_ai_location
        self.processor_id = processor_id or doc_ai.google_doc_ai_processor_id
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

        if not all([self.project_id, self.processor_id]):
            raise ValueError(
                "Google Document AI requires project_id and processor_id. "
                "Set GOOGLE_DOC_AI_PROJECT_ID and GOOGLE_DOC_AI_PROCESSOR_ID in .env"
            )

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google Document AI client."""
        opts = ClientOptions(
            api_endpoint=f"{self.location}-documentai.googleapis.com"
        )
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
        self.processor_name = self.client.processor_path(
            self.project_id, self.location, self.processor_id
        )
        logger.info(
            f"Google Document AI client initialized. Processor: {self.processor_name}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type from file extension."""
        extension = Path(file_path).suffix.lstrip(".").lower()
        return self.MIME_TYPES.get(extension, "application/pdf")

    def _build_process_options(self) -> documentai.ProcessOptions:
        """Build ProcessOptions with LayoutConfig + ChunkingConfig for
        structured, layout-aware output."""
        chunking_config = documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
            chunk_size=self.chunk_size,
            include_ancestor_headings=True,
        )
        layout_config = documentai.ProcessOptions.LayoutConfig(
            chunking_config=chunking_config,
        )
        return documentai.ProcessOptions(layout_config=layout_config)

    def _extract_content_from_chunks(self, document: documentai.Document) -> str:
        """Extract and combine content from chunked_document.
        Falls back to document.text if no chunks are available."""
        chunked_doc = document.chunked_document
        if chunked_doc and chunked_doc.chunks:
            chunk_contents = [
                chunk.content for chunk in chunked_doc.chunks if chunk.content
            ]
            logger.info(
                f"Extracted {len(chunk_contents)} layout-aware chunks from document"
            )
            return self.CHUNK_SEPARATOR.join(chunk_contents)

        logger.warning("No chunks in chunked_document, falling back to document.text")
        return document.text or ""

    # ------------------------------------------------------------------
    # Scanned-PDF detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_scanned_pdf(file_path: str, sample_pages: int = 5) -> bool:
        """Return True when most sampled pages contain no extractable text,
        indicating the PDF is image-only / scanned."""
        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            return False

        try:
            reader = PdfReader(str(path))
        except Exception:
            return False

        total = len(reader.pages)
        if total == 0:
            return False

        pages_to_check = min(sample_pages, total)
        empty_pages = 0
        for i in range(pages_to_check):
            text = (reader.pages[i].extract_text() or "").strip()
            if len(text) < 30:
                empty_pages += 1

        return empty_pages / pages_to_check >= 0.6

    # ------------------------------------------------------------------
    # Synchronous processing
    # ------------------------------------------------------------------

    def _process_document(self, file_path: str) -> str:
        """Synchronously process a document through Document AI Layout Parser.

        Returns structured text extracted from layout-aware chunks.
        """
        raw = Path(file_path).read_bytes()

        if len(raw) > ONLINE_PROCESSING_MAX_BYTES:
            raise ValueError(
                f"File size ({len(raw) / 1024 / 1024:.1f} MB) exceeds the "
                f"Document AI online processing limit of 20 MB."
            )

        mime_type = self._detect_mime_type(file_path)
        raw_document = documentai.RawDocument(content=raw, mime_type=mime_type)
        process_options = self._build_process_options()

        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
            process_options=process_options,
        )

        logger.info(f"Sending {file_path} to Document AI ({mime_type})...")
        result = self.client.process_document(request=request)
        text = self._extract_content_from_chunks(result.document)
        logger.info(f"Document AI returned {len(text)} characters from {file_path}")
        return text

    def ocr(self, file_path: str) -> str:
        """Process the file via Document AI and return extracted text (sync)."""
        return self._process_document(file_path)

    async def ocr_async(self, file_path: str) -> str:
        """Process the file via Document AI and return extracted text (async).
        Runs the blocking gRPC call in a thread pool."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, self._process_document, file_path
            )

    # ------------------------------------------------------------------
    # High-level orchestrator
    # ------------------------------------------------------------------

    def process(self, file_path: str, output_dir: str | None = None) -> str:
        """If *file_path* is a scanned PDF, OCR it and return the path to a
        markdown file containing the extracted text.  Otherwise return
        *file_path* unchanged so the caller can proceed with docling."""
        if not self.is_scanned_pdf(file_path):
            logger.info(f"{file_path} is not a scanned PDF, skipping OCR")
            return file_path

        text = self.ocr(file_path)

        if output_dir is None:
            out = Path(file_path).with_suffix(".md")
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{Path(file_path).stem}.md"

        out.write_text(text, encoding="utf-8")
        logger.info(f"OCR result saved to {out}")
        return str(out)

    async def process_async(self, file_path: str, output_dir: str | None = None) -> str:
        """Async version of process()."""
        if not self.is_scanned_pdf(file_path):
            logger.info(f"{file_path} is not a scanned PDF, skipping OCR")
            return file_path

        text = await self.ocr_async(file_path)

        if output_dir is None:
            out = Path(file_path).with_suffix(".md")
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{Path(file_path).stem}.md"

        out.write_text(text, encoding="utf-8")
        logger.info(f"OCR result saved to {out}")
        return str(out)
