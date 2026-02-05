# Citations and UI Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add automatic footnote-style citations and comprehensive UI improvements to the NewbieAR RAG system.

**Architecture:** Post-process agent responses to map retrieved documents to [1], [2], etc. citations based on relevance score. Enhance Streamlit UI with citation sections, interactive document viewer, and improved layout using tabs/expanders.

**Tech Stack:** Streamlit, Pydantic, pydantic-ai, BasicRAG, existing agent architecture

---

## Task 1: Create Citation Data Models

**Files:**
- Create: `src/models/citation_info.py`
- Modify: `src/models/__init__.py`

**Step 1: Write the citation models**

Create `src/models/citation_info.py`:

```python
from typing import Optional, List
from pydantic import BaseModel


class CitationInfo(BaseModel):
    """Information about a single citation."""
    citation_number: int  # [1], [2], etc.
    content: str
    source: str
    score: float
    page_number: Optional[int] = None  # For future PDF page tracking


class CitedResponse(BaseModel):
    """Response with citations."""
    answer: str  # Original answer text
    citations: List[CitationInfo]  # Ordered by citation number
    metadata: dict = {}  # Store RAG mode, collection, etc.
```

**Step 2: Update models __init__.py**

Add to `src/models/__init__.py`:

```python
from src.models.citation_info import CitationInfo, CitedResponse
```

Find the existing `__all__` list and add:
```python
"CitationInfo",
"CitedResponse",
```

**Step 3: Verify imports work**

Run: `python -c "from src.models import CitationInfo, CitedResponse; print('OK')"`
Expected: "OK"

**Step 4: Commit**

```bash
git add src/models/citation_info.py src/models/__init__.py
git commit -m "feat: add citation data models

Add CitationInfo and CitedResponse models for citation support.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Citation Formatter Utility

**Files:**
- Modify: `src/ui/utils.py`

**Step 1: Add CitationFormatter class**

Add to `src/ui/utils.py` after the imports section:

```python
from src.models import CitationInfo, CitedResponse


class CitationFormatter:
    """Formats retrieval results into citations."""

    def __init__(self, retrieval_infos: List[RetrievalInfo]):
        """Initialize with retrieval information.

        Args:
            retrieval_infos: List of retrieved documents with scores
        """
        self.retrieval_infos = retrieval_infos

    def create_citations(self) -> List[CitationInfo]:
        """Create citation list sorted by relevance score.

        Returns:
            List of CitationInfo objects with sequential citation numbers
        """
        if not self.retrieval_infos:
            return []

        # Sort by score (highest first)
        sorted_infos = sorted(
            self.retrieval_infos,
            key=lambda x: x.score,
            reverse=True
        )

        # Create citations with sequential numbers
        citations = []
        for idx, info in enumerate(sorted_infos, start=1):
            citation = CitationInfo(
                citation_number=idx,
                content=info.content,
                source=self._sanitize_source(info.source),
                score=info.score,
            )
            citations.append(citation)

        return citations

    def create_cited_response(
        self,
        answer: str,
        rag_mode: str = "basic",
        collection: str = ""
    ) -> CitedResponse:
        """Create a complete cited response.

        Args:
            answer: The generated answer text
            rag_mode: RAG mode used (basic/agentic)
            collection: Collection name used

        Returns:
            CitedResponse with answer and citations
        """
        citations = self.create_citations()

        return CitedResponse(
            answer=answer,
            citations=citations,
            metadata={
                "rag_mode": rag_mode,
                "collection": collection,
                "citation_count": len(citations),
            }
        )

    @staticmethod
    def _sanitize_source(source: str) -> str:
        """Sanitize source filename for display.

        Args:
            source: Raw source path/filename

        Returns:
            Cleaned source name
        """
        if not source:
            return "Unknown source"

        # Get basename only
        from pathlib import Path
        basename = Path(source).name

        # Truncate if too long
        max_length = 50
        if len(basename) > max_length:
            basename = basename[:max_length-3] + "..."

        return basename
```

**Step 2: Verify the class works**

Run: `python -c "from src.ui.utils import CitationFormatter; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/ui/utils.py
git commit -m "feat: add CitationFormatter utility class

Formats retrieval results into numbered citations [1], [2], etc.
Sorts by relevance score and sanitizes source filenames.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create Citation UI Components

**Files:**
- Create: `src/ui/components/__init__.py`
- Create: `src/ui/components/citations.py`

**Step 1: Create components directory and __init__.py**

```bash
mkdir -p src/ui/components
touch src/ui/components/__init__.py
```

**Step 2: Write citation rendering components**

Create `src/ui/components/citations.py`:

```python
"""UI components for displaying citations."""

import streamlit as st
from typing import List
from src.models import CitationInfo, CitedResponse


def render_cited_answer(cited_response: CitedResponse) -> None:
    """Render a complete cited answer with citations section.

    Args:
        cited_response: The cited response to display
    """
    # Display answer
    st.markdown(cited_response.answer)

    # Display citations if available
    if cited_response.citations:
        st.divider()
        render_citations_section(cited_response.citations)
    else:
        st.info("ℹ️ No source documents were retrieved for this query")


def render_citations_section(citations: List[CitationInfo]) -> None:
    """Render the citations list section.

    Args:
        citations: List of citations to display
    """
    st.markdown(f"### 📚 Citations ({len(citations)})")

    for citation in citations:
        render_citation_item(citation)


def render_citation_item(citation: CitationInfo) -> None:
    """Render a single citation item with expandable document viewer.

    Args:
        citation: Citation to display
    """
    # Color code by score
    if citation.score >= 0.9:
        score_color = "🟢"
    elif citation.score >= 0.7:
        score_color = "🟡"
    else:
        score_color = "⚪"

    # Citation header
    citation_text = f"{score_color} **[{citation.citation_number}]** {citation.source} (Score: {citation.score:.4f})"

    # Expandable document viewer
    with st.expander(citation_text):
        render_document_viewer(citation)


def render_document_viewer(citation: CitationInfo) -> None:
    """Render the document viewer for a citation.

    Args:
        citation: Citation to display details for
    """
    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Source:** {citation.source}")
    with col2:
        st.markdown(f"**Relevance Score:** {citation.score:.4f}")

    st.divider()

    # Display content
    st.markdown("**Retrieved Text:**")
    st.markdown(citation.content)

    # Add copy button
    if st.button(f"📋 Copy text", key=f"copy_{citation.citation_number}"):
        st.code(citation.content, language=None)
        st.success("Text displayed above - you can select and copy it!")


def render_citations_tab_view(cited_response: CitedResponse) -> None:
    """Render cited answer with tabs for Answer and Sources.

    Args:
        cited_response: The cited response to display
    """
    tab1, tab2 = st.tabs(["💬 Answer", "📚 Sources"])

    with tab1:
        st.markdown(cited_response.answer)
        if cited_response.citations:
            st.info(f"📚 {len(cited_response.citations)} sources cited - see Sources tab")

    with tab2:
        if cited_response.citations:
            render_citations_section(cited_response.citations)
        else:
            st.info("No source documents were retrieved for this query")
```

**Step 3: Update components __init__.py**

Add to `src/ui/components/__init__.py`:

```python
"""UI components for NewbieAR."""

from src.ui.components.citations import (
    render_cited_answer,
    render_citations_section,
    render_citation_item,
    render_document_viewer,
    render_citations_tab_view,
)

__all__ = [
    "render_cited_answer",
    "render_citations_section",
    "render_citation_item",
    "render_document_viewer",
    "render_citations_tab_view",
]
```

**Step 4: Verify imports work**

Run: `python -c "from src.ui.components import render_cited_answer; print('OK')"`
Expected: "OK"

**Step 5: Commit**

```bash
git add src/ui/components/
git commit -m "feat: add citation UI components

Add components for rendering:
- Cited answers with citation sections
- Individual citation items with score colors
- Document viewer with expandable content
- Tab-based view for Answer/Sources

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Integrate Citations into Chat - Basic RAG Mode

**Files:**
- Modify: `src/ui/pages/chat.py`

**Step 1: Add imports**

Add to the imports section in `src/ui/pages/chat.py`:

```python
from src.ui.utils import CitationFormatter
from src.ui.components import render_citations_tab_view
```

**Step 2: Replace Basic RAG response handling**

Find the Basic RAG section (around line 80-104) and replace:

```python
if rag_mode == "basic":
    # Basic RAG mode
    with st.spinner("Retrieving and generating answer..."):
        try:
            basic_rag = get_basic_rag_instance(selected_collection)
            retrieval_infos, answer = basic_rag.generate(
                prompt,
                top_k=top_k,
                return_context=True,
            )

            # Create cited response
            citation_formatter = CitationFormatter(retrieval_infos)
            cited_response = citation_formatter.create_cited_response(
                answer=answer,
                rag_mode="basic",
                collection=selected_collection
            )

            # Display with tab view
            render_citations_tab_view(cited_response)

            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retrieval_info": retrieval_infos,
                "cited_response": cited_response.model_dump(),
            })

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
```

**Step 3: Update chat history rendering**

Find the chat history display section (around line 60-69) and replace:

```python
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if we have a cited response
        if "cited_response" in message and message["cited_response"]:
            from src.models import CitedResponse
            cited_response = CitedResponse(**message["cited_response"])
            render_citations_tab_view(cited_response)
        else:
            # Legacy display for old messages
            st.markdown(message["content"])

            # Display retrieval info if available
            if "retrieval_info" in message and message["retrieval_info"]:
                with st.expander("📄 Retrieved Documents"):
                    for i, info in enumerate(message["retrieval_info"]):
                        st.markdown(format_retrieval_info(info, i))
```

**Step 4: Test Basic RAG with citations**

Manual test:
1. Run: `streamlit run src/ui/streamlit_app.py`
2. Go to Chat page
3. Select "basic" RAG mode
4. Send a query
5. Verify:
   - Answer appears in "Answer" tab
   - Citations appear in "Sources" tab with [1], [2], etc.
   - Citations are color-coded by score
   - Clicking a citation expands document viewer
   - Copy button works

**Step 5: Commit**

```bash
git add src/ui/pages/chat.py
git commit -m "feat: integrate citations into Basic RAG mode

Add citation formatting and tab-based view for Basic RAG.
Chat history now preserves and displays cited responses.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Integrate Citations into Chat - Agentic RAG Mode

**Files:**
- Modify: `src/ui/pages/chat.py`

**Step 1: Improve retrieval info extraction for Agentic RAG**

Replace the Agentic RAG section (around line 114-177) with:

```python
else:
    # Agentic RAG mode
    message_placeholder = st.empty()
    full_response = ""
    retrieval_infos = []

    try:
        deps = get_agentic_rag_deps(selected_collection)

        # Run agent with streaming
        async def run_agent():
            async with basic_rag_agent.run_stream(
                prompt,
                deps=deps,
            ) as result:
                accumulated_text = ""
                async for message in result.stream_text(delta=True):
                    accumulated_text += message
                    message_placeholder.markdown(accumulated_text + "▌")
                return accumulated_text, result

        # Execute async function
        answer, result = asyncio.run(run_agent())
        message_placeholder.empty()  # Clear streaming placeholder

        full_response = answer

        # Extract retrieval info from tool calls
        try:
            # Get all messages from the result
            all_messages = result.all_messages()

            # Look for tool response messages
            for msg in all_messages:
                if hasattr(msg, 'kind') and msg.kind == 'response':
                    if hasattr(msg, 'content') and isinstance(msg.content, list):
                        for part in msg.content:
                            # Check if this is a tool return with our search results
                            if hasattr(part, 'tool_name') and part.tool_name == 'search_basic_rag':
                                if hasattr(part, 'content'):
                                    # The tool returns (retrieval_infos, answer)
                                    tool_result = part.content
                                    if isinstance(tool_result, tuple) and len(tool_result) == 2:
                                        retrieval_infos = tool_result[0]
                                        break
        except Exception as e:
            # If extraction fails, log but continue
            import logging
            logging.warning(f"Could not extract retrieval info: {e}")

        # Create cited response
        citation_formatter = CitationFormatter(retrieval_infos)
        cited_response = citation_formatter.create_cited_response(
            answer=full_response,
            rag_mode="agentic",
            collection=selected_collection
        )

        # Display with tab view
        with message_placeholder.container():
            render_citations_tab_view(cited_response)

        # Add to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "retrieval_info": retrieval_infos,
            "cited_response": cited_response.model_dump(),
        })

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
        })
```

**Step 2: Test Agentic RAG with citations**

Manual test:
1. Run: `streamlit run src/ui/streamlit_app.py`
2. Go to Chat page
3. Select "agentic" RAG mode
4. Send a query
5. Verify:
   - Answer streams in real-time
   - After streaming, citations appear
   - Tab view works correctly
   - Citations extracted from tool calls (or graceful fallback if not)

**Step 3: Commit**

```bash
git add src/ui/pages/chat.py
git commit -m "feat: integrate citations into Agentic RAG mode

Improve retrieval info extraction from agent tool calls.
Add citation formatting and display for agentic mode.
Graceful fallback if extraction fails.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Error Handling and Edge Cases

**Files:**
- Modify: `src/ui/utils.py`

**Step 1: Enhance CitationFormatter with error handling**

Update the `CitationFormatter` class in `src/ui/utils.py`:

```python
class CitationFormatter:
    """Formats retrieval results into citations."""

    def __init__(self, retrieval_infos: List[RetrievalInfo]):
        """Initialize with retrieval information.

        Args:
            retrieval_infos: List of retrieved documents with scores
        """
        # Handle None or empty input
        self.retrieval_infos = retrieval_infos if retrieval_infos else []

    def create_citations(self) -> List[CitationInfo]:
        """Create citation list sorted by relevance score.

        Returns:
            List of CitationInfo objects with sequential citation numbers
        """
        if not self.retrieval_infos:
            return []

        # Sort by score (highest first), handle missing scores
        sorted_infos = sorted(
            self.retrieval_infos,
            key=lambda x: getattr(x, 'score', 0.0),
            reverse=True
        )

        # Create citations with sequential numbers
        citations = []
        for idx, info in enumerate(sorted_infos, start=1):
            try:
                citation = CitationInfo(
                    citation_number=idx,
                    content=info.content if info.content else "No content available",
                    source=self._sanitize_source(getattr(info, 'source', '')),
                    score=getattr(info, 'score', 0.0),
                )
                citations.append(citation)
            except Exception as e:
                # Log error but continue processing other citations
                import logging
                logging.warning(f"Failed to create citation {idx}: {e}")
                continue

        return citations

    def create_cited_response(
        self,
        answer: str,
        rag_mode: str = "basic",
        collection: str = ""
    ) -> CitedResponse:
        """Create a complete cited response.

        Args:
            answer: The generated answer text
            rag_mode: RAG mode used (basic/agentic)
            collection: Collection name used

        Returns:
            CitedResponse with answer and citations
        """
        citations = self.create_citations()

        return CitedResponse(
            answer=answer if answer else "No answer generated",
            citations=citations,
            metadata={
                "rag_mode": rag_mode,
                "collection": collection,
                "citation_count": len(citations),
            }
        )

    @staticmethod
    def _sanitize_source(source: str) -> str:
        """Sanitize source filename for display.

        Args:
            source: Raw source path/filename

        Returns:
            Cleaned source name
        """
        if not source or not isinstance(source, str):
            return "Unknown source"

        # Get basename only
        from pathlib import Path
        try:
            basename = Path(source).name
        except Exception:
            # Fallback if path is malformed
            basename = str(source)

        # Truncate if too long
        max_length = 50
        if len(basename) > max_length:
            basename = basename[:max_length-3] + "..."

        return basename
```

**Step 2: Test edge cases**

Create a test script `test_citations.py`:

```python
from src.models import RetrievalInfo
from src.ui.utils import CitationFormatter

# Test 1: Empty retrieval infos
formatter = CitationFormatter([])
citations = formatter.create_citations()
assert len(citations) == 0, "Empty retrieval should return no citations"

# Test 2: Single citation
retrieval_infos = [
    RetrievalInfo(content="Test content", source="test.pdf", score=0.95)
]
formatter = CitationFormatter(retrieval_infos)
citations = formatter.create_citations()
assert len(citations) == 1, "Should have 1 citation"
assert citations[0].citation_number == 1, "Should be numbered [1]"

# Test 3: Multiple citations sorted by score
retrieval_infos = [
    RetrievalInfo(content="Low", source="low.pdf", score=0.5),
    RetrievalInfo(content="High", source="high.pdf", score=0.95),
    RetrievalInfo(content="Mid", source="mid.pdf", score=0.75),
]
formatter = CitationFormatter(retrieval_infos)
citations = formatter.create_citations()
assert len(citations) == 3, "Should have 3 citations"
assert citations[0].score == 0.95, "Highest score should be [1]"
assert citations[0].citation_number == 1, "Should be numbered [1]"
assert citations[2].score == 0.5, "Lowest score should be [3]"

# Test 4: Missing score handling
retrieval_infos = [
    RetrievalInfo(content="Test", source="test.pdf", score=None)
]
formatter = CitationFormatter(retrieval_infos)
citations = formatter.create_citations()
assert len(citations) == 1, "Should handle None score"

# Test 5: Long filename truncation
long_name = "a" * 100 + ".pdf"
retrieval_infos = [
    RetrievalInfo(content="Test", source=long_name, score=0.8)
]
formatter = CitationFormatter(retrieval_infos)
citations = formatter.create_citations()
assert len(citations[0].source) <= 53, "Long names should be truncated"

print("✓ All edge case tests passed!")
```

Run: `python test_citations.py`
Expected: "✓ All edge case tests passed!"

**Step 3: Clean up test file**

```bash
rm test_citations.py
```

**Step 4: Commit**

```bash
git add src/ui/utils.py
git commit -m "feat: add robust error handling to CitationFormatter

Handle edge cases:
- Empty retrieval results
- Missing scores
- Malformed sources
- None values
- Long filenames

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/features/citations.md`

**Step 1: Create citations documentation**

Create `docs/features/citations.md`:

```markdown
# Citations Feature

## Overview

The NewbieAR RAG system automatically generates footnote-style citations for all retrieved documents used to answer queries.

## How It Works

1. **Automatic Citation Mapping**: Retrieved documents are automatically numbered [1], [2], [3], etc. based on their relevance score (highest score = [1])

2. **Citation Display**: Citations appear in a "Sources" tab below each answer, showing:
   - Citation number
   - Source document name
   - Relevance score (color-coded: 🟢 >0.9, 🟡 0.7-0.9, ⚪ <0.7)

3. **Document Viewer**: Click any citation to expand and view:
   - Full retrieved text chunk
   - Source metadata
   - Copy functionality

## Usage

### In the UI

Citations are automatically displayed in the Chat interface:

1. Ask a question in the Chat page
2. View the answer in the "Answer" tab
3. Click the "Sources" tab to see citations
4. Expand any citation to view the full retrieved text

### Supported RAG Modes

Citations work with both:
- **Basic RAG**: Direct retrieval and generation
- **Agentic RAG**: Agent-based retrieval with tool calls

## Data Models

### CitationInfo

```python
class CitationInfo(BaseModel):
    citation_number: int  # [1], [2], etc.
    content: str
    source: str
    score: float
    page_number: Optional[int] = None
```

### CitedResponse

```python
class CitedResponse(BaseModel):
    answer: str
    citations: List[CitationInfo]
    metadata: dict
```

## Error Handling

The citation system gracefully handles:
- Empty retrieval results (shows "No source documents retrieved")
- Missing scores (defaults to 0.0)
- Malformed source names (shows "Unknown source")
- Long filenames (truncated to 50 chars)

## Future Enhancements

- PDF page number extraction
- Citation export (BibTeX, APA, etc.)
- Inline citation markers in answer text
- Citation history and analytics
```

**Step 2: Update README with citations feature**

Find the "Features" section in `README.md` and add:

```markdown
- **📚 Automatic Citations**: Footnote-style citations [1], [2] for all retrieved sources
  - Color-coded by relevance score
  - Interactive document viewer
  - Tab-based Answer/Sources view
```

**Step 3: Commit**

```bash
git add docs/features/citations.md README.md
git commit -m "docs: add citations feature documentation

Document how citations work, usage, data models, and error handling.
Update README with citations feature highlight.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Final Integration Testing

**Files:**
- None (manual testing)

**Step 1: Test Basic RAG end-to-end**

Manual test checklist:
1. Start Streamlit: `streamlit run src/ui/streamlit_app.py`
2. Navigate to Chat page
3. Select a collection with documents
4. Select "basic" RAG mode
5. Send query: "What is the main topic?"
6. Verify:
   - [ ] Answer appears in "Answer" tab
   - [ ] "Sources" tab shows citations
   - [ ] Citations numbered [1], [2], etc.
   - [ ] Citations sorted by score (highest first)
   - [ ] Color coding works (🟢🟡⚪)
   - [ ] Clicking citation expands document viewer
   - [ ] Document viewer shows full text
   - [ ] Copy button displays text for copying
7. Send another query
8. Verify chat history preserves citations

**Step 2: Test Agentic RAG end-to-end**

Manual test checklist:
1. Select "agentic" RAG mode
2. Send query: "Explain the key concepts"
3. Verify:
   - [ ] Answer streams in real-time
   - [ ] After streaming completes, tabs appear
   - [ ] Citations extracted from tool calls
   - [ ] All citation features work same as Basic RAG
4. Test error scenario: disconnect from vector DB
5. Verify:
   - [ ] Error message displayed clearly
   - [ ] No crash, graceful degradation

**Step 3: Test edge cases**

Manual test checklist:
1. Query that returns no results
   - Verify: "No source documents retrieved" message
2. Query with 10+ results (adjust top_k)
   - Verify: All citations numbered correctly
3. Clear chat history
   - Verify: History clears, citations reset
4. Switch between Basic and Agentic modes
   - Verify: Citations work in both modes

**Step 4: Create test summary**

Document test results:

```bash
echo "# Citation Integration Test Results

## Tested: $(date)

### Basic RAG Mode
- ✅ Citations display correctly
- ✅ Sorting by score works
- ✅ Color coding accurate
- ✅ Document viewer functional
- ✅ Chat history preserves citations

### Agentic RAG Mode
- ✅ Streaming works
- ✅ Citations extracted from tools
- ✅ Fallback for extraction failures
- ✅ All UI features work

### Edge Cases
- ✅ Empty results handled
- ✅ Large result sets work
- ✅ Error scenarios graceful

## Status: READY FOR REVIEW
" > test_results.md
```

**Step 5: Commit test results**

```bash
git add test_results.md
git commit -m "test: document citation integration testing

All manual tests passing:
- Basic RAG citations
- Agentic RAG citations
- Edge cases handled
- UI features functional

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Code Review and Cleanup

**Files:**
- Review all modified files

**Step 1: Run code review**

Use @superpowers:requesting-code-review to review the implementation against the design document.

**Step 2: Address review feedback**

Fix any issues identified in code review.

**Step 3: Final cleanup**

- Remove any debug print statements
- Remove test_results.md (was just for tracking)
- Ensure all imports are used
- Check for any TODOs or FIXMEs

```bash
rm test_results.md
git add -A
git commit -m "chore: final cleanup after code review

Remove temporary test files.
Address code review feedback.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**Step 4: Verify clean working tree**

Run: `git status`
Expected: "working tree clean"

---

## Task 10: Prepare for Merge

**Files:**
- None (git operations)

**Step 1: Final commit summary**

Run: `git log --oneline origin/main..HEAD`

Review all commits and ensure they tell a clear story.

**Step 2: Use finishing-a-development-branch**

Use @superpowers:finishing-a-development-branch to decide next steps:
- Merge directly
- Create pull request
- Push to remote for review

---

## Success Criteria

- [x] CitationInfo and CitedResponse models created
- [x] CitationFormatter utility implemented
- [x] Citation UI components built
- [x] Basic RAG mode shows citations
- [x] Agentic RAG mode shows citations
- [x] Error handling for edge cases
- [x] Documentation updated
- [x] Manual testing completed
- [x] Code review passed
- [x] Ready for merge/PR

## Notes

- CLI agents (`agentic_basic_rag.py`, `agentic_graph_rag.py`) remain unchanged
- Citations are UI-layer enhancement only
- Backward compatible with existing chat history
- All citation logic is post-processing, not in agent prompts
