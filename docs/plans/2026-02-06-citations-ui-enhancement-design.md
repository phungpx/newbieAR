# Citations and UI Enhancement Design

**Date:** 2026-02-06
**Status:** Approved
**Scope:** Add automatic footnote-style citations and comprehensive UI improvements to RAG system

## Overview

This design enhances the NewbieAR RAG system with automatic citation generation and improved UI for better source attribution and user experience. Citations will be automatically mapped to retrieved documents without requiring LLM awareness, and the UI will provide an enhanced viewing experience with better layout and interactive document exploration.

## Architecture Overview

### Core Enhancement Strategy

The enhancement consists of two main layers:

1. **Data Layer (Agent Enhancement)** - Add post-processing to inject citation markers into agent responses. The `search_basic_rag` and `search_graphiti` tools continue returning retrieval info, with citation formatting applied as a wrapper.

2. **Presentation Layer (UI Enhancement)** - Redesign Streamlit chat interface with three key components:
   - **Main Answer Panel** - Answer with inline citation markers [1], [2]
   - **Citations Section** - Formatted list of sources at the bottom
   - **Document Viewer Modal** - Click citations to see full retrieved chunks

### Key Design Decisions

- **Automatic Citation Mapping**: Retrieved documents mapped to [1], [2], etc. based on relevance score (highest score = [1])
- **Non-Invasive Agent Changes**: Agents don't need prompt or behavior changes - citation formatting is a wrapper
- **Backward Compatibility**: CLI versions of agents continue working unchanged; only UI integration adds citations

## Data Structures & Components

### Enhanced Data Models

**CitationInfo Model** (extends RetrievalInfo):
```python
class CitationInfo(BaseModel):
    citation_number: int  # [1], [2], etc.
    content: str
    source: str
    score: float
    page_number: Optional[int] = None  # For future PDF page tracking
```

**CitedResponse Model**:
```python
class CitedResponse(BaseModel):
    answer: str  # Original answer text
    citations: List[CitationInfo]  # Ordered by citation number
    metadata: dict  # Store RAG mode, collection, etc.
```

### New Components

1. **CitationFormatter (utils.py)** - Helper class that:
   - Takes retrieval_infos and generates CitationInfo list
   - Sorts by relevance score
   - Assigns sequential citation numbers [1], [2], etc.

2. **render_cited_answer() (new component)** - Streamlit component that:
   - Displays answer in main panel
   - Renders "📚 Citations" section below with formatted list
   - Each citation shows: `[1] Source: document.pdf (Score: 0.95)`

3. **render_document_viewer() (new component)** - Modal/expander that:
   - Triggered when user clicks a citation
   - Shows full retrieved text with highlighting
   - Displays metadata (source, score, chunk position)

## Data Flow & Integration

### End-to-End Flow

1. **User submits query** → Streamlit chat input

2. **RAG Processing:**
   - **Basic RAG**: `basic_rag.generate()` returns `(retrieval_infos, answer)`
   - **Agentic RAG**: `basic_rag_agent.run_stream()` returns result, extract retrieval_infos from tool calls

3. **Citation Processing** (new step):
   ```python
   citation_formatter = CitationFormatter(retrieval_infos)
   cited_response = citation_formatter.create_cited_response(answer)
   # cited_response now has: answer, citations list [1], [2], etc.
   ```

4. **UI Rendering:**
   - Main chat message shows `cited_response.answer`
   - Below answer: `render_citations_section(cited_response.citations)`
   - User clicks `[1]` → triggers `render_document_viewer(citation_1)`

### Key Integration Points

- **chat.py line 92-104** (Basic RAG): Add citation formatting before displaying answer
- **chat.py line 136-169** (Agentic RAG): Add citation formatting and improve retrieval_infos extraction
- **utils.py**: Add `CitationFormatter` class and new rendering functions
- **Session state**: Store `cited_response` instead of just `answer` for history

### Streaming Consideration

For agentic mode, stream the answer for real-time feedback, but add citations after streaming completes (they appear below the fully-streamed answer).

## UI Design & User Experience

### Enhanced Chat Layout

```
┌─────────────────────────────────────┐
│ 💬 Assistant Response               │
│                                     │
│ [Answer text appears here...]       │
│                                     │
├─────────────────────────────────────┤
│ 📚 Citations (3)                    │
│                                     │
│ [1] document1.pdf (Score: 0.95)     │
│ [2] document2.pdf (Score: 0.87)     │
│ [3] document3.pdf (Score: 0.82)     │
│                                     │
├─────────────────────────────────────┤
│ 📄 Retrieved Documents ▼            │
│ (Expandable - shows all chunks)     │
└─────────────────────────────────────┘
```

### Interactive Features

1. **Citations Section** - Compact list using `st.info()` or custom styled markdown
   - Each citation is clickable (using Streamlit buttons or expanders)
   - Shows source filename and relevance score
   - Color-coded by score (green: >0.9, yellow: 0.7-0.9, gray: <0.7)

2. **Document Viewer** - When clicking a citation:
   - Opens in `st.expander()` or nested container
   - Shows full chunk text with `st.code()` or `st.markdown()` for readability
   - Displays metadata: source, score, chunk size
   - "Copy text" button for easy reference

3. **Layout Improvements:**
   - Use `st.tabs()` to separate "Answer" and "Sources" for cleaner view
   - Or use columns: `col1` for answer, `col2` for citations sidebar
   - Add visual dividers with `st.divider()` between sections

## Error Handling & Edge Cases

### Error Scenarios

1. **No retrieval results** (empty retrieval_infos):
   - Display answer without citations section
   - Show info message: "No source documents were retrieved for this query"
   - Agent can still generate answer from general knowledge

2. **Citation extraction failure** (agentic mode):
   - Fallback: Show answer without citations
   - Log warning for debugging
   - Display: "Citations unavailable - retrieval info could not be extracted"

3. **Malformed source filenames**:
   - Sanitize display names (truncate long paths, show basename only)
   - Handle missing source field gracefully with "Unknown source"

4. **Duplicate sources**:
   - Multiple chunks from same document get separate citations [1], [2]
   - Each citation shows chunk position if available
   - Example: `[1] doc.pdf (chunk 1/5)`, `[2] doc.pdf (chunk 3/5)`

5. **Score edge cases**:
   - Handle None or missing scores with default display
   - Very low scores (<0.5) shown with warning indicator

### Testing Strategy

- **Unit tests**: CitationFormatter with various retrieval_infos inputs
- **Integration tests**: Both RAG modes with citation generation
- **UI tests**: Manual verification of citation clicking, expansion, layout
- **Edge case tests**: Empty results, single result, 10+ results

### Backwards Compatibility

- Existing CLI agents (`agentic_basic_rag.py`, `agentic_graph_rag.py`) remain unchanged
- Only UI integration adds citation layer
- Old chat history (without citations) displays gracefully

## Implementation Notes

### Files to Modify

1. **src/models/citation_info.py** (new) - CitationInfo and CitedResponse models
2. **src/ui/utils.py** - Add CitationFormatter class
3. **src/ui/components/citations.py** (new) - render_cited_answer, render_citations_section, render_document_viewer
4. **src/ui/pages/chat.py** - Integrate citation formatting and new UI components

### Files to Keep Unchanged

- **src/agents/agentic_basic_rag.py** - CLI agent stays as-is
- **src/agents/agentic_graph_rag.py** - CLI agent stays as-is
- **src/retrieval/** - No changes to retrieval logic

## Success Criteria

1. ✅ Automatic citation numbers [1], [2], etc. appear for all retrieved documents
2. ✅ Citations displayed in clean, formatted list below answers
3. ✅ Clicking citations shows full retrieved chunk content
4. ✅ UI has better visual separation and organization
5. ✅ Both Basic RAG and Agentic RAG modes support citations
6. ✅ CLI agents continue working without changes
7. ✅ Error cases handled gracefully with appropriate fallbacks
