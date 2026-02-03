# Hierarchical Chunking Strategy Design

**Date:** 2026-02-03
**Status:** Approved

## Overview

Enhance `src/ingestion/chunk_document.py` to support hierarchical chunking as an alternative strategy to the existing hybrid chunking approach. This design adds a strategy pattern that allows users to choose between chunking methods while maintaining a unified interface.

## Requirements

- Add hierarchical chunking strategy based on tested implementation in `examples/hierarchical_chunking.ipynb`
- Maintain backward compatibility with existing hybrid chunking
- Preserve unified interface for both strategies
- Extend metadata to capture hierarchical-specific information

## Architecture

### Strategy Pattern

The `DocChunker` class will support multiple chunking strategies through a `strategy` parameter:

- **Single entry point:** `DocChunker` remains the main class
- **Strategy selection:** Constructor parameter `strategy` with values `"hybrid"` (default) or `"hierarchical"`
- **Conditional initialization:** Different chunker instances based on strategy
- **Shared interface:** Both strategies expose the same `chunk_document()` method

### Backward Compatibility

Existing code continues working unchanged:
- `strategy` defaults to `"hybrid"`
- All existing parameters maintain their current behavior
- Return type remains `tuple[list[ChunkInfo], str]`

## Design Details

### 1. Constructor Parameters

```python
class DocChunker:
    def __init__(
        self,
        strategy: str = "hybrid",           # NEW: chunking strategy
        tokenizer_name: str = MODEL_ID,
        max_tokens: int = MAX_CHUNKED_TOKENS,  # hybrid only
        merge_peers: bool = True,              # hybrid only
        always_emit_headings: bool = False,    # hybrid only
        merge_list_items: bool = True,         # hierarchical only
        output_dir: str = None,
    ):
```

**Strategy-Aware Parameters:**
- Hybrid-only: `max_tokens`, `merge_peers`, `always_emit_headings`
- Hierarchical-only: `merge_list_items`
- Shared: `tokenizer_name`, `output_dir`
- Unused parameters are silently ignored (no warnings)

### 2. Initialization Logic

**Hybrid Strategy:**
```python
if strategy == "hybrid":
    self.chunker = HybridChunker(
        tokenizer=self.tokenizer,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
        always_emit_headings=always_emit_headings,
    )
```

**Hierarchical Strategy:**
```python
elif strategy == "hierarchical":
    self.serializer_provider = ImgPlaceholderSerializerProvider()
    self.chunker = HierarchicalChunker(
        serializer_provider=self.serializer_provider,
        merge_list_items=merge_list_items,
    )
```

**Serializer Configuration:**
- Uses fixed `ImgPlaceholderSerializerProvider` with image placeholders
- `merge_list_items=True` by default
- No additional customization options (keeping it simple)

### 3. chunk_document() Method

**Unified Processing:**
```python
def chunk_document(self, file_path: str) -> tuple[list[ChunkInfo], str]:
    # Load document (same for both)
    document = self.loader.convert(source=file_path).document

    # Chunk document (strategy-aware API call)
    if self.strategy == "hierarchical":
        chunk_iter = self.chunker.chunk(dl_doc=document)
    else:
        chunk_iter = self.chunker.chunk(document)

    # Process chunks
    for i, chunk in enumerate(chunk_iter):
        # Extract common fields
        text = chunk.text
        contextualized_text = self.chunker.contextualize(chunk=chunk)
        text_tokens = self.tokenizer.count_tokens(text)
        contextualized_tokens = self.tokenizer.count_tokens(contextualized_text)

        # Extract strategy-specific metadata
        if self.strategy == "hierarchical":
            doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
            doc_items_labels = [it.label.value for it in chunk.meta.doc_items]
        else:
            doc_items_refs = None
            doc_items_labels = None

        # Create unified ChunkInfo
        chunk_info = ChunkInfo(...)
```

### 4. ChunkInfo Model Extension

**New Optional Fields:**
```python
class ChunkInfo(BaseModel):
    # Existing fields
    chunk_id: int
    text: str
    text_tokens: int
    contextualized_text: str
    contextualized_tokens: int
    filename: str
    mimetype: str

    # NEW: Hierarchical metadata (optional)
    doc_items_refs: Optional[list[str]] = None
    doc_items_labels: Optional[list[str]] = None
```

**Field Semantics:**
- `doc_items_refs`: Document item references (e.g., `['#/texts/122', '#/tables/0']`)
- `doc_items_labels`: Item type labels (e.g., `['TEXT', 'TABLE']`)
- Both fields are `None` for hybrid chunks
- Both fields are populated arrays for hierarchical chunks

### 5. Error Handling

**Strategy Validation:**
```python
VALID_STRATEGIES = {"hybrid", "hierarchical"}

if strategy not in VALID_STRATEGIES:
    raise ValueError(
        f"Invalid strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}"
    )
```

**Required Imports:**
```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownParams
```

**Logging Enhancement:**
```python
logger.info(f"Chunking document using {self.strategy} strategy...")
```

## Usage Examples

### Programmatic Usage

**Hybrid (existing behavior):**
```python
chunker = DocChunker(output_dir="./chunks")
chunks, path = chunker.chunk_document("doc.pdf")
```

**Hierarchical (new):**
```python
chunker = DocChunker(
    strategy="hierarchical",
    merge_list_items=True,
    output_dir="./chunks"
)
chunks, path = chunker.chunk_document("doc.pdf")
```

### CLI Usage

**Hybrid:**
```bash
python chunk_document.py --file_path doc.pdf --output_dir ./chunks
```

**Hierarchical:**
```bash
python chunk_document.py --file_path doc.pdf --output_dir ./chunks --strategy hierarchical
```

## Implementation Tasks

1. Update `src/models/chunkinfo.py` to add optional fields
2. Update `src/ingestion/chunk_document.py`:
   - Add imports for hierarchical chunker
   - Add strategy parameter and validation
   - Implement conditional chunker initialization
   - Update chunk_document() method for both strategies
   - Add CLI argument for strategy (optional)
3. Test both strategies with sample documents
4. Update documentation/README if needed

## Trade-offs & Decisions

**Why strategy pattern over separate classes?**
- Simpler API for users (one class to learn)
- Easier to maintain (shared loading and saving logic)
- Better backward compatibility

**Why optional fields over separate models?**
- Downstream code doesn't break
- Single serialization format
- Easy to ignore extra metadata if not needed

**Why fixed serializer defaults?**
- YAGNI - avoid over-engineering
- Can add customization later if needed
- Keeps API surface small

## Future Enhancements

- Add CLI `--strategy` argument support
- Support custom serializer providers
- Add more chunking strategies (e.g., semantic, fixed-size)
- Expose more hierarchical-specific configuration options
