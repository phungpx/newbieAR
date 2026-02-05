"""UI components for displaying tool calls."""

import streamlit as st
from typing import List
from src.models import ToolCallInfo


def render_tool_calls(tool_calls: List[ToolCallInfo]) -> None:
    """Render tool calls with execution details.

    Args:
        tool_calls: List of tool calls to display
    """
    if not tool_calls:
        return

    st.markdown(f"### 🔧 Tool Calls ({len(tool_calls)})")

    for i, tool_call in enumerate(tool_calls):
        render_tool_call_item(tool_call, index=i)


def render_tool_call_item(tool_call: ToolCallInfo, index: int = 0) -> None:
    """Render a single tool call item.

    Args:
        tool_call: Tool call to display
        index: Index for unique keys
    """
    # Status indicator
    if tool_call.status == "success":
        status_icon = "✅"
        status_color = "green"
    elif tool_call.status == "error":
        status_icon = "❌"
        status_color = "red"
    else:
        status_icon = "⏳"
        status_color = "yellow"

    # Tool call header
    header_text = f"{status_icon} **{tool_call.tool_name}**"
    if tool_call.execution_time:
        header_text += f" ({tool_call.execution_time:.2f}s)"

    # Expandable details
    with st.expander(header_text):
        # Status
        st.markdown(f"**Status:** {tool_call.status}")

        # Arguments
        if tool_call.arguments:
            st.markdown("**Arguments:**")
            st.json(tool_call.arguments)

        # Result preview (truncated if large)
        if tool_call.result is not None:
            st.markdown("**Result:**")
            result_str = str(tool_call.result)
            if len(result_str) > 1000:
                st.text(result_str[:1000] + "...")
                with st.expander("View full result"):
                    st.text(result_str)
            else:
                st.text(result_str)

        # Execution time
        if tool_call.execution_time:
            st.markdown(f"**Execution Time:** {tool_call.execution_time:.2f}s")


def render_tool_calls_compact(tool_calls: List[ToolCallInfo]) -> None:
    """Render tool calls in a compact format.

    Args:
        tool_calls: List of tool calls to display
    """
    if not tool_calls:
        return

    # Show as badges
    for tool_call in tool_calls:
        status_icon = (
            "✅"
            if tool_call.status == "success"
            else "❌" if tool_call.status == "error" else "⏳"
        )
        st.markdown(f"{status_icon} `{tool_call.tool_name}`", unsafe_allow_html=True)
