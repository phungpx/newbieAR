from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel


class Chunk(BaseModel):
    content: str
    metadata: dict | None = None


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


def get_document_loader(file_path: str) -> BaseLoader:
    extension = Path(file_path).suffix
    extension = extension.lower()
    if extension not in DOCUMENT_LOADER_MAPPING:
        raise ValueError(f"Unsupported file extension: {extension}")
    return DOCUMENT_LOADER_MAPPING[extension](file_path)


def chunk_document(
    file_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
) -> list[Chunk]:
    document_loader = get_document_loader(file_path)
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = document_loader.load()
    chunks = text_splitter.split_documents(documents)
    return [
        Chunk(
            content=chunk.page_content,
            metadata={
                "source": Path(file_path).name,
                "chunk_id": index,
            },
        )
        for index, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    file_path = "/Users/phung.pham/Documents/PHUNGPX/deepeval_exploration/RAG/data/theranos_legacy.txt"
    chunks = chunk_document(file_path, chunk_size=100, chunk_overlap=10)
    print(chunks)
