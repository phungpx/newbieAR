from tqdm.notebook import tqdm
from datasets import load_dataset
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.settings import ProjectSettings
from src.deps._qdrant_client import QdrantVectorStore

settings = ProjectSettings()

dataset = load_dataset("atitaarora/qdrant_doc", split="train")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

qdrant_vector_store = QdrantVectorStore(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

documents = [
    Document(
        page_content=document["text"],
        metadata={"source": document["source"]},
    )
    for document in tqdm(dataset)
]

chunks: list[Document] = []
for document in documents:
    chunks += text_splitter.split_documents([document])

chunk_contents, chunk_metadatas = [], []

for chunk in chunks:
    if hasattr(chunk, "page_content") and hasattr(chunk, "metadata"):
        chunk_contents.append(chunk.page_content)
        chunk_metadatas.append(chunk.metadata)
    else:
        raise ValueError(
            "Some documents do not have 'page_content' or 'metadata' attributes."
        )

# Uses FastEmbed - https://qdrant.tech/documentation/fastembed/
# To generate embeddings for the documents
# The default model is `BAAI/bge-small-en-v1.5`
qdrant_vector_store.add_embeddings(
    collection_name=settings.qdrant_collection_name,
    metadata=chunk_metadatas,
    documents=chunk_contents,
)
