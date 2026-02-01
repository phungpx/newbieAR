from enum import Enum


class VectordbIngestionTasks(Enum):
    UPLOAD_RAW_FILE = "upload_raw_file"
    CHUNK_RAW_FILE = "chunk_raw_file"
    EMBED_AND_INDEX = "embed_and_index"
    GENERATE_MARKDOWN_PREVIEW = "generate_markdown_preview"
