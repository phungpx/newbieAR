# BasicRAG Agent & GraphRAG Refinements Design

**Date:** 2026-02-05
**Status:** Approved

## Overview

Create a new conversational agent for BasicRAG retrieval and refine the existing GraphRAG agent's error handling. Both agents will follow consistent patterns using pydantic-ai for orchestration.

## Goals

1. Create `agentic_basic_rag.py` with interactive chat interface
2. Improve error handling and robustness in `agentic_graph_rag.py`
3. Maintain architectural consistency across both agents
4. Provide excellent user experience with Rich UI and streaming

## Architecture

### File Structure

**New files:**
- `src/agents/agentic_basic_rag.py` - BasicRAG agent implementation
- `src/prompts/basic_rag_instruction.py` - System prompt for BasicRAG agent

**Modified files:**
- `src/agents/agentic_graph_rag.py` - Enhanced error handling
- `src/prompts/__init__.py` - Export new prompt

### Shared Patterns

Both agents share:
- Pydantic-AI Agent with dependencies dataclass
- Tool-based architecture with async tools
- Interactive async chat loop with Rich UI
- Markdown streaming via Live display
- Conversation history persistence
- File-based logging via loguru

## BasicRAG Agent Design

### Dependencies Structure

```python
@dataclass
class BasicRAGDependencies:
    basic_rag: BasicRAG
    top_k: int = 5
```

### Tool Design

Single unified tool `search_basic_rag`:

```python
@basic_rag_agent.tool
async def search_basic_rag(
    ctx: RunContext[BasicRAGDependencies],
    query: str
) -> tuple[list[RetrievalInfo], str]:
    """
    Search the vector database and generate an answer.

    Args:
        ctx: Context containing BasicRAG instance and config
        query: The search query

    Returns:
        Tuple of (retrieval_infos, generated_answer)
    """
```

**Tool behavior:**
1. Validate query is non-empty
2. Call `basic_rag.generate(query, top_k, return_context=True)`
3. Log execution to file (not console)
4. Handle errors with `ModelRetry` for transient failures
5. Return both retrieval context and generated answer

### System Prompt

```python
BASIC_RAG_AGENT_INSTRUCTION = """You are a helpful assistant with access to a vector-based document retrieval system.

When the user asks a question:
1. Use your search tool to query the document database
2. The tool returns retrieved documents with relevance scores and a generated answer
3. Provide clear, accurate responses based on the retrieved information
4. Cite sources when referencing specific documents
5. Be honest when the retrieved documents don't contain sufficient information to answer the question

If search results are insufficient or unclear, you can:
- Ask the user for clarification to refine the search
- Acknowledge the limitations of the available documents
- Suggest rephrasing the question

Always prioritize accuracy over completeness - it's better to admit uncertainty than to hallucinate information."""
```

### Interactive CLI Interface

**Main loop structure:**
```python
async def main():
    # Initialize components
    basic_rag = BasicRAG(qdrant_collection_name=args.collection_name)
    console = Console()
    messages: list[ModelMessage] = []

    # Welcome panel
    console.print(Panel("BasicRAG Knowledge Agent..."))

    while True:
        # Non-blocking user input
        user_input = await get_user_input(console)

        # Exit handling
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        # Stream response with Live markdown display
        with Live(Spinner("dots", "Thinking...")) as live:
            try:
                async with basic_rag_agent.run_stream(
                    user_input,
                    message_history=messages,
                    deps=deps
                ) as result:
                    accumulated_text = ""
                    async for message in result.stream_text(delta=True):
                        accumulated_text += message
                        live.update(Markdown(accumulated_text))

                messages.extend(result.all_messages())

            except Exception as e:
                logger.exception(f"Stream error: {e}")
                live.update(Panel(f"[bold red]Error:[/] {str(e)}", border_style="red"))
```

**CLI arguments:**
- `--collection_name` (required) - Qdrant collection to query
- `--top_k` (optional, default=5) - Number of documents to retrieve

## GraphRAG Agent Refinements

### Error Handling Improvements

**1. Tool-level validation and error handling:**

```python
@graphiti_agent.tool
async def search_graphiti(ctx: RunContext[GraphitiDependencies], query: str):
    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    logger.info(f"Tool executing search for: {query}")

    try:
        results = await graph_retrieval.retrieve(query, num_results=ctx.deps.top_k)
        node_infos, edge_infos, episode_infos = results

        logger.debug(f"Retrieved: {len(node_infos)} nodes, {len(edge_infos)} edges")

        # Handle empty results gracefully
        if not any([node_infos, edge_infos, episode_infos]):
            logger.warning(f"No results for query: {query}")
            return [], [], []

        return node_infos, edge_infos, episode_infos

    except asyncio.TimeoutError:
        logger.error("Search timed out")
        raise ModelRetry("Search timed out. Try a simpler query.")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise ModelRetry(f"Database connection failed. Please try again.")
    except Exception as e:
        logger.exception(f"Graphiti search failed: {e}")
        raise ModelRetry(f"Search encountered an error. Try rephrasing your query.")
```

**2. Stream-level error recovery:**

```python
with Live(Spinner("dots", "Thinking...")) as live:
    try:
        async with graphiti_agent.run_stream(...) as result:
            accumulated_text = ""
            async for message in result.stream_text(delta=True):
                accumulated_text += message
                live.update(Markdown(accumulated_text))

        messages.extend(result.all_messages())

    except ModelRetry as e:
        # Expected retry - show friendly message
        live.update(Panel(f"[yellow]{str(e)}[/yellow]", border_style="yellow"))
    except Exception as e:
        # Unexpected error
        logger.exception(f"Stream error: {e}")
        live.update(Panel(f"[bold red]Error:[/] {str(e)}", border_style="red"))
```

**3. Input validation at entry point:**

```python
# Before calling agent
user_input = await get_user_input(console)

# Validate
if not user_input or not user_input.strip():
    console.print("[yellow]Please enter a question.[/yellow]")
    continue

if len(user_input) > 1000:
    console.print("[yellow]Query too long. Please keep it under 1000 characters.[/yellow]")
    continue
```

### Error Categories Handled

1. **Empty/invalid queries** - Validation before processing
2. **Database connection failures** - Retry with user-friendly message
3. **Timeout errors** - Suggest simpler query
4. **Empty search results** - Return gracefully, let agent explain
5. **Unexpected exceptions** - Log details, show generic error to user
6. **Stream interruptions** - Catch and display without crashing loop

## Code Organization

### Approach: Self-contained agents (no shared utilities)

**Rationale:**
- Each agent remains independent and easy to modify
- Minimal code duplication (~50 lines)
- No coupling between agent modules
- Can evolve separately as requirements change
- Can refactor later if more agents are added

**Duplicated code:**
- `get_openai_model()` - Model initialization
- `get_user_input()` - Async input handling
- Logger configuration

**Consistency guidelines:**
- Loguru: `rotation="10 MB", level="DEBUG"`
- Agent retries: `retries=2`
- Exit keywords: `["exit", "quit", "bye"]`
- Spinner text: `"Thinking..."`
- User prompt color: `[bold green]`
- Assistant label color: `[bold purple]`

## Success Criteria

**BasicRAG Agent:**
- ✓ Implements tool-based architecture matching GraphRAG
- ✓ Provides interactive async chat with streaming
- ✓ Returns structured retrieval + generation results
- ✓ Handles errors gracefully without crashing
- ✓ Logs to file for debugging
- ✓ Accepts collection_name and top_k via CLI

**GraphRAG Refinements:**
- ✓ Validates all user inputs
- ✓ Handles connection/timeout errors with retry
- ✓ Manages empty results gracefully
- ✓ Catches and displays stream errors
- ✓ Logs detailed errors to file
- ✓ Maintains conversation loop despite errors

## Implementation Notes

1. Keep existing `BasicRAG` class unchanged - agent wraps it
2. Use same pydantic-ai patterns as GraphRAG
3. Test error scenarios: empty queries, connection failures, timeouts
4. Verify streaming works smoothly with Rich Live display
5. Ensure conversation history persists across turns
6. Check logging doesn't interfere with UI rendering

## Future Enhancements (Out of Scope)

- Shared utilities module if more agents are added
- Conversation export/save functionality
- Multi-turn query refinement
- Hybrid search combining BasicRAG + GraphRAG
- Configurable streaming vs non-streaming modes
