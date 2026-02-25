# Design: Docling Paragraph-Level Chunking for Synthesis

**Date:** 2026-02-25
**Branch:** features/update_synthesis
**Status:** Approved

## Problem

`synthesize.py` currently uses deepeval's built-in `generate_goldens_from_docs()` with `ContextConstructionConfig` (fixed `chunk_size=1024`, `chunk_overlap=20`). This naive character-count chunking splits text at arbitrary boundaries, degrading context quality for synthetic golden generation — especially for structured documents like research papers.

## Goal

Pre-chunk documents using docling's structure-aware `HierarchicalChunker` (paragraph boundaries + token cap), then feed the resulting contexts into deepeval's `generate_goldens_from_contexts()`. This preserves semantic paragraph boundaries and respects document structure.

## Architecture

### Before

```
file_path
  → deepeval internal loader (character-split, chunk_size=1024)
  → generate_goldens_from_docs()
  → goldens
```

### After

```
file_path
  → DocumentChunker(strategy='hierarchical')   # docling HierarchicalChunker
  → build_contexts_from_doc()                  # sliding window grouping
  → generate_goldens_from_contexts()           # deepeval synthesizer
  → goldens
```

## Components

### 1. `src/synthesis/utils.py` — new helper function

**Add:** `build_contexts_from_doc(file_path, window_size=3, stride=1) -> list[list[str]]`

- Instantiates `DocumentChunker(strategy='hierarchical')` with no `output_dir` (no JSON side-effects)
- Calls `chunk_document(file_path)` to get `list[ChunkInfo]`
- Extracts `chunk.text` from each `ChunkInfo`
- Applies a sliding window: window of `window_size` consecutive chunk texts, advancing by `stride`
- Returns `list[list[str]]` — deepeval's expected context format

**Window defaults:** `window_size=3`, `stride=1`

### 2. `src/synthesis/synthesize.py` — `__main__` block

**Remove:**
- `ContextConstructionConfig` (import + instantiation + usage)
- `context_construction_config` variable
- `embeder` variable (only used by `ContextConstructionConfig`)

**Add import:**
- `from src.synthesis.utils import build_contexts_from_doc`

**Replace** `generate_goldens_from_docs()` call:

```python
# OLD
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[str(file_path)],
    include_expected_output=True,
    context_construction_config=context_construction_config,
    max_goldens_per_context=1,
)

# NEW
contexts = build_contexts_from_doc(str(file_path), window_size=3, stride=1)
logger.info(f"Built {len(contexts)} contexts from {file_path.name}")
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
    source_files=[str(file_path)],
)
```

**Keep unchanged:** `GPTModel`, `Synthesizer`, `FiltrationConfig`, `EvolutionConfig`, `StylingConfig`, `STYLING_CONFIG`, `TOPIC`, argparse setup.

## Data Flow

```
ChunkInfo(text, contextualized_text, ...)
    ↓  [extract .text]
["paragraph 1 text", "paragraph 2 text", ...]
    ↓  [sliding window, size=3, stride=1]
[["p1","p2","p3"], ["p2","p3","p4"], ["p3","p4","p5"], ...]
    ↓  [generate_goldens_from_contexts]
[Golden(input, expected_output, context=[...]), ...]
```

## Trade-offs

| Aspect | Before | After |
|---|---|---|
| Chunk granularity | Character count (1024 chars) | Paragraph/structural boundary |
| Context quality | May split mid-sentence | Semantically complete paragraphs |
| Startup cost | Lightweight | Loads HuggingFace tokenizer (sentence-transformers/all-MiniLM-L6-v2) |
| Code complexity | Single API call | Two-step: chunk → synthesize |
| Reusability | deepeval-only | `build_contexts_from_doc` reusable elsewhere |

## Files Changed

- `src/synthesis/utils.py` — add `build_contexts_from_doc()`
- `src/synthesis/synthesize.py` — remove `ContextConstructionConfig`/`embeder`, replace synthesis call
