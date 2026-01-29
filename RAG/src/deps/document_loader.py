from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    Docx2txtLoader,
)
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

DOCUMENT_LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".mdx": TextLoader,
    ".csv": CSVLoader,
    ".web": WebBaseLoader,
}


class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader: BaseLoader = self.get_loader(file_path)

    def get_loader(self, file_path: str) -> BaseLoader:
        if Path(file_path).suffix not in DOCUMENT_LOADER_MAPPING:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")
        return DOCUMENT_LOADER_MAPPING[Path(file_path).suffix](file_path)

    def load(self) -> list[Document]:
        return self.loader.load()
