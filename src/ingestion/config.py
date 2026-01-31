from pathlib import Path

# Model and tokenization settings
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNKED_TOKENS = 1024

# Directory settings
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
CHUNKS_DIR = OUTPUT_DIR / "chunks"

# Chunker settings
CHUNKER_CONFIG = {
    "tokenizer_name": MODEL_ID,
    "max_tokens": MAX_CHUNKED_TOKENS,
    "merge_peers": True,  # Merges small adjacent items (like list items) into one chunk
    "always_emit_headings": False,
}

# Supported file formats
SUPPORTED_FORMATS = [
    ".pdf",
    ".docx",
    ".txt",
    ".doc",
    ".pptx",
    ".md",
]
