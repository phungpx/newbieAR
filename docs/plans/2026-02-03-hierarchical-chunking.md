# Hierarchical Chunking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add hierarchical chunking strategy to DocChunker alongside existing hybrid chunking.

**Architecture:** Strategy pattern with conditional initialization - single DocChunker class accepts strategy parameter ("hybrid" or "hierarchical"), initializes appropriate chunker implementation, and produces unified ChunkInfo output with optional hierarchical metadata.

**Tech Stack:** docling, docling-core, transformers, pydantic

---

## Task 1: Extend ChunkInfo Model

**Files:**
- Modify: `src/models/chunkinfo.py:1-12`

**Step 1: Add optional fields to ChunkInfo**

```python
from typing import Optional
from pydantic import BaseModel


class ChunkInfo(BaseModel):
    chunk_id: int
    text: str
    text_tokens: int
    contextualized_text: str
    contextualized_tokens: int
    filename: str
    mimetype: str

    # Hierarchical chunking metadata (optional)
    doc_items_refs: Optional[list[str]] = None
    doc_items_labels: Optional[list[str]] = None
```

**Step 2: Verify model still works**

Run: `python -c "from src.models import ChunkInfo; c = ChunkInfo(chunk_id=0, text='test', text_tokens=1, contextualized_text='test', contextualized_tokens=1, filename='f', mimetype='m'); print(c)"`

Expected: ChunkInfo instance printed successfully

**Step 3: Commit**

```bash
git add src/models/chunkinfo.py
git commit -m "feat: extend ChunkInfo with optional hierarchical metadata fields"
```

---

## Task 2: Add Hierarchical Chunker Imports

**Files:**
- Modify: `src/ingestion/chunk_document.py:1-14`

**Step 1: Add hierarchical chunker imports**

Add after existing imports (around line 8):

```python
from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownParams
```

**Step 2: Verify imports work**

Run: `python -c "from src.ingestion.chunk_document import *; print('Imports successful')"`

Expected: "Imports successful"

**Step 3: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: add hierarchical chunker imports"
```

---

## Task 3: Add Strategy Constants and Serializer Provider

**Files:**
- Modify: `src/ingestion/chunk_document.py:12-15`

**Step 1: Add strategy constants**

Add after `MAX_CHUNKED_TOKENS = 1024` (line 13):

```python
VALID_STRATEGIES = {"hybrid", "hierarchical"}


class ImgPlaceholderSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider for hierarchical chunking with image placeholders."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )
```

**Step 2: Verify syntax**

Run: `python -c "from src.ingestion.chunk_document import VALID_STRATEGIES, ImgPlaceholderSerializerProvider; print(VALID_STRATEGIES)"`

Expected: `{'hybrid', 'hierarchical'}`

**Step 3: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: add strategy constants and serializer provider"
```

---

## Task 4: Update DocChunker Constructor Signature

**Files:**
- Modify: `src/ingestion/chunk_document.py:16-35`

**Step 1: Add strategy parameter and merge_list_items**

Update `__init__` signature (line 17):

```python
def __init__(
    self,
    strategy: str = "hybrid",
    tokenizer_name: str = MODEL_ID,
    max_tokens: int = MAX_CHUNKED_TOKENS,
    merge_peers: bool = True,
    always_emit_headings: bool = False,
    merge_list_items: bool = True,
    output_dir: str = None,
):
```

**Step 2: Add strategy validation**

Add at the beginning of `__init__` body (after line 26):

```python
if strategy not in VALID_STRATEGIES:
    raise ValueError(
        f"Invalid strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}"
    )
self.strategy = strategy
```

**Step 3: Verify validation works**

Run: `python -c "from src.ingestion.chunk_document import DocChunker; chunker = DocChunker(strategy='invalid')"`

Expected: ValueError with message about invalid strategy

**Step 4: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: add strategy parameter with validation"
```

---

## Task 5: Implement Conditional Chunker Initialization

**Files:**
- Modify: `src/ingestion/chunk_document.py:25-35`

**Step 1: Replace chunker initialization with strategy-aware logic**

Replace lines 29-34 with:

```python
if strategy == "hybrid":
    self.chunker = HybridChunker(
        tokenizer=self.tokenizer,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
        always_emit_headings=always_emit_headings,
    )
elif strategy == "hierarchical":
    self.serializer_provider = ImgPlaceholderSerializerProvider()
    self.chunker = HierarchicalChunker(
        serializer_provider=self.serializer_provider,
        merge_list_items=merge_list_items,
    )
```

**Step 2: Verify hybrid chunker still initializes**

Run: `python -c "from src.ingestion.chunk_document import DocChunker; chunker = DocChunker(); print(type(chunker.chunker).__name__)"`

Expected: `HybridChunker`

**Step 3: Verify hierarchical chunker initializes**

Run: `python -c "from src.ingestion.chunk_document import DocChunker; chunker = DocChunker(strategy='hierarchical'); print(type(chunker.chunker).__name__)"`

Expected: `HierarchicalChunker`

**Step 4: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: implement conditional chunker initialization by strategy"
```

---

## Task 6: Update chunk_document Method for Strategy-Aware Chunking

**Files:**
- Modify: `src/ingestion/chunk_document.py:37-76`

**Step 1: Update chunking call to be strategy-aware**

Replace line 46 with:

```python
if self.strategy == "hierarchical":
    chunk_iter = self.chunker.chunk(dl_doc=document)
else:
    chunk_iter = self.chunker.chunk(document)
```

**Step 2: Update logging to include strategy**

Replace line 44 with:

```python
logger.info(f"Chunking document using {self.strategy} strategy...")
```

**Step 3: Verify syntax**

Run: `python -c "from src.ingestion.chunk_document import DocChunker; print('Syntax OK')"`

Expected: `Syntax OK`

**Step 4: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: add strategy-aware chunking and logging"
```

---

## Task 7: Extract Hierarchical Metadata in chunk_document

**Files:**
- Modify: `src/ingestion/chunk_document.py:48-62`

**Step 1: Add hierarchical metadata extraction**

After line 52 (contextualized_tokens calculation), add:

```python
# Extract strategy-specific metadata
if self.strategy == "hierarchical":
    doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
    doc_items_labels = [it.label.value for it in chunk.meta.doc_items]
else:
    doc_items_refs = None
    doc_items_labels = None
```

**Step 2: Update ChunkInfo construction**

Update ChunkInfo instantiation (lines 53-61) to include new fields:

```python
chunk_info = ChunkInfo(
    chunk_id=i,
    text=chunk.text,
    text_tokens=text_tokens,
    contextualized_text=contextualized_text,
    contextualized_tokens=contextualized_tokens,
    filename=chunk.meta.origin.filename,
    mimetype=chunk.meta.origin.mimetype,
    doc_items_refs=doc_items_refs,
    doc_items_labels=doc_items_labels,
)
```

**Step 3: Verify syntax**

Run: `python -c "from src.ingestion.chunk_document import DocChunker; print('Syntax OK')"`

Expected: `Syntax OK`

**Step 4: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: extract and include hierarchical metadata in ChunkInfo"
```

---

## Task 8: Test Hybrid Strategy (Backward Compatibility)

**Files:**
- Test: `src/ingestion/chunk_document.py` (manual test)

**Step 1: Find a test document**

Run: `find examples -name "*.pdf" -o -name "*.docx" | head -1`

Expected: Path to a test document (or use a URL like in the notebook)

**Step 2: Test hybrid chunking**

Run: `python src/ingestion/chunk_document.py --file_path "<path_from_step_1>" --output_dir /tmp/test_hybrid`

Expected:
- Log shows "Chunking document using hybrid strategy..."
- Chunks saved successfully
- Output JSON has `doc_items_refs: null` and `doc_items_labels: null`

**Step 3: Verify output structure**

Run: `python -c "import json; data = json.load(open('/tmp/test_hybrid/<filename>.json')); print('doc_items_refs' in data[0], data[0]['doc_items_refs'])"`

Expected: `True None`

**Step 4: Commit test results**

```bash
git add .
git commit -m "test: verify hybrid strategy backward compatibility"
```

---

## Task 9: Test Hierarchical Strategy

**Files:**
- Test: `src/ingestion/chunk_document.py` (manual test)

**Step 1: Create test script**

Create temporary test file:

```python
# /tmp/test_hierarchical.py
from src.ingestion.chunk_document import DocChunker

chunker = DocChunker(
    strategy="hierarchical",
    merge_list_items=True,
    output_dir="/tmp/test_hierarchical"
)

# Use URL from notebook example
chunks, output_path = chunker.chunk_document("https://arxiv.org/pdf/1706.03762")
print(f"Created {len(chunks)} chunks")
print(f"Saved to {output_path}")

# Check first chunk with table
for chunk in chunks:
    if chunk.doc_items_labels and 'TABLE' in chunk.doc_items_labels:
        print(f"\nFound table in chunk {chunk.chunk_id}")
        print(f"  refs: {chunk.doc_items_refs}")
        print(f"  labels: {chunk.doc_items_labels}")
        break
```

**Step 2: Run hierarchical chunking test**

Run: `cd /Users/phung.pham/Documents/PHUNGPX/deepeval_exploration && python /tmp/test_hierarchical.py`

Expected:
- Log shows "Chunking document using hierarchical strategy..."
- Chunks created with populated `doc_items_refs` and `doc_items_labels`
- Table chunks identified correctly

**Step 3: Verify hierarchical metadata populated**

Run: `python -c "import json; data = json.load(open('/tmp/test_hierarchical/1706.03762.json')); chunk = next(c for c in data if c.get('doc_items_labels') and 'TABLE' in c['doc_items_labels']); print('Found:', chunk['doc_items_labels'])"`

Expected: List containing 'TABLE' label

**Step 4: Clean up and commit**

```bash
rm /tmp/test_hierarchical.py
git add .
git commit -m "test: verify hierarchical strategy with metadata extraction"
```

---

## Task 10: Add CLI Argument for Strategy (Optional Enhancement)

**Files:**
- Modify: `src/ingestion/chunk_document.py:93-103`

**Step 1: Add --strategy CLI argument**

Update argparse section (around line 96-99):

```python
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument(
    "--strategy",
    type=str,
    default="hybrid",
    choices=["hybrid", "hierarchical"],
    help="Chunking strategy to use (default: hybrid)"
)
args = parser.parse_args()

chunker = DocChunker(strategy=args.strategy, output_dir=args.output_dir)
```

**Step 2: Test CLI with hybrid**

Run: `python src/ingestion/chunk_document.py --file_path "<test_doc>" --output_dir /tmp/cli_test --strategy hybrid`

Expected: Works as before

**Step 3: Test CLI with hierarchical**

Run: `python src/ingestion/chunk_document.py --file_path "<test_doc>" --output_dir /tmp/cli_test --strategy hierarchical`

Expected: Uses hierarchical strategy

**Step 4: Test CLI help**

Run: `python src/ingestion/chunk_document.py --help`

Expected: Shows --strategy argument with choices

**Step 5: Commit**

```bash
git add src/ingestion/chunk_document.py
git commit -m "feat: add CLI argument for strategy selection"
```

---

## Task 11: Final Integration Test

**Files:**
- Test: End-to-end validation

**Step 1: Test programmatic API for both strategies**

Create and run test script:

```python
# /tmp/final_test.py
from src.ingestion.chunk_document import DocChunker

# Test 1: Hybrid (default, backward compat)
print("Test 1: Hybrid chunking (backward compat)")
chunker1 = DocChunker(output_dir="/tmp/final_hybrid")
assert chunker1.strategy == "hybrid"
print("✓ Hybrid chunker created with default strategy")

# Test 2: Explicit hybrid
print("\nTest 2: Explicit hybrid strategy")
chunker2 = DocChunker(strategy="hybrid", max_tokens=512, output_dir="/tmp/final_hybrid2")
assert chunker2.strategy == "hybrid"
print("✓ Hybrid chunker created with explicit strategy")

# Test 3: Hierarchical
print("\nTest 3: Hierarchical strategy")
chunker3 = DocChunker(strategy="hierarchical", merge_list_items=True, output_dir="/tmp/final_hier")
assert chunker3.strategy == "hierarchical"
print("✓ Hierarchical chunker created")

# Test 4: Invalid strategy
print("\nTest 4: Invalid strategy handling")
try:
    DocChunker(strategy="invalid")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "Invalid strategy" in str(e)
    print("✓ Invalid strategy rejected correctly")

print("\n✅ All integration tests passed!")
```

Run: `python /tmp/final_test.py`

Expected: All tests pass

**Step 2: Verify imports in downstream code won't break**

Run: `python -c "from src.models import ChunkInfo; from src.ingestion.chunk_document import DocChunker; print('All imports working')"`

Expected: "All imports working"

**Step 3: Clean up test files**

Run: `rm -rf /tmp/test_* /tmp/final_* /tmp/cli_test`

**Step 4: Final commit**

```bash
git add .
git commit -m "test: add final integration tests for both strategies"
```

---

## Testing Checklist

- [ ] ChunkInfo model accepts optional fields (Task 1)
- [ ] Hierarchical imports work (Task 2)
- [ ] Strategy validation rejects invalid values (Task 4)
- [ ] Hybrid chunker initializes by default (Task 5)
- [ ] Hierarchical chunker initializes with strategy param (Task 5)
- [ ] Hybrid strategy produces chunks with null metadata (Task 8)
- [ ] Hierarchical strategy produces chunks with populated metadata (Task 9)
- [ ] CLI accepts --strategy argument (Task 10)
- [ ] Backward compatibility maintained (Task 11)

## Verification Commands

After implementation, verify everything works:

```bash
# 1. Import check
python -c "from src.ingestion.chunk_document import DocChunker; from src.models import ChunkInfo"

# 2. Hybrid default
python -c "from src.ingestion.chunk_document import DocChunker; c = DocChunker(); assert c.strategy == 'hybrid'"

# 3. Hierarchical explicit
python -c "from src.ingestion.chunk_document import DocChunker; c = DocChunker(strategy='hierarchical'); assert c.strategy == 'hierarchical'"

# 4. Invalid strategy
python -c "from src.ingestion.chunk_document import DocChunker; DocChunker(strategy='invalid')" 2>&1 | grep "Invalid strategy"
```

## Notes

- No existing tests to update (no test directory found)
- Manual testing required using example documents or URLs
- Consider adding pytest tests in future for regression prevention
- The notebook in `examples/hierarchical_chunking.ipynb` serves as reference implementation
