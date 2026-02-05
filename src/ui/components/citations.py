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
