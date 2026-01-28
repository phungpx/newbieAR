from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.deps.vector_stores import QdrantVectorStore
from src.deps.embeddings import EmbeddingClient
from src.settings import settings

vector_store = QdrantVectorStore(
    uri=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

logger.info(f"Creating collection {settings.qdrant_collection_name}")
try:
    vector_store.create_collection(
        collection_name=settings.qdrant_collection_name,
        embedding_size=settings.embedding_dimensions,
        distance="cosine",
    )
    logger.info(f"Created collection {settings.qdrant_collection_name}")
except Exception as e:
    logger.error(f"Error creating collection {settings.qdrant_collection_name}: {e}")
    raise e

embedding_client = EmbeddingClient(
    model_name=settings.embedding_model_name,
    batch_size=settings.embedding_batch_size,
    model_dim=settings.embedding_dimensions,
)

dataset = load_dataset("atitaarora/qdrant_doc", split="train")

all_documents = [
    Document(page_content=document["text"], metadata={"source": document["source"]})
    for document in tqdm(dataset)
]

logger.info(f"Loaded {len(all_documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

chunked_documents = []
for document in tqdm(all_documents):
    chunked_documents += text_splitter.split_documents([document])
logger.info(f"Chunked {len(chunked_documents)} documents")

embeddings = embedding_client.embed(chunked_documents)
logger.info(f"Embedded {len(embeddings)} documents")

vector_store.add_embeddings(
    collection_name=settings.qdrant.qdrant_collection_name,
    embeddings=embeddings,
    payloads=[document.metadata for document in chunked_documents],
    batch_size=2,
)
logger.info(
    f"Added {len(embeddings)} documents to collection {settings.qdrant.qdrant_collection_name}"
)
