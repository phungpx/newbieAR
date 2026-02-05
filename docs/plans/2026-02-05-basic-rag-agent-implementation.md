# BasicRAG Agent & GraphRAG Refinements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a conversational agent for BasicRAG retrieval with interactive chat and refine GraphRAG agent error handling.

**Architecture:** Pydantic-AI agent with tool-based architecture wrapping existing BasicRAG class, async chat loop with Rich UI streaming, matching GraphRAG patterns for consistency.

**Tech Stack:** pydantic-ai, Rich (UI), loguru (logging), asyncio, BasicRAG (existing retrieval)

---

## Task 1: Create BasicRAG System Prompt

**Files:**
- Create: `src/prompts/basic_rag_instruction.py`
- Modify: `src/prompts/__init__.py`

**Step 1: Create the system prompt file**

Create `src/prompts/basic_rag_instruction.py`:

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

**Step 2: Update prompts __init__.py**

Read the current file:
```bash
cat src/prompts/__init__.py
```

Add import and export to `src/prompts/__init__.py`:

```python
from .generation import RAG_GENERATION_PROMPT
from .filtering import FilterTemplate
from .evolution import EvolutionTemplate
from .conversational_evolution import ConversationalEvolutionTemplate
from .synthesizer import SynthesizerTemplate
from .graphiti_instruction import GRAPHITI_AGENT_INSTRUCTION
from .basic_rag_instruction import BASIC_RAG_AGENT_INSTRUCTION

__all__ = [
    "RAG_GENERATION_PROMPT",
    "FilterTemplate",
    "EvolutionTemplate",
    "ConversationalEvolutionTemplate",
    "SynthesizerTemplate",
    "GRAPHITI_AGENT_INSTRUCTION",
    "BASIC_RAG_AGENT_INSTRUCTION",
]
```

**Step 3: Verify imports work**

Run: `uv run python -c "from src.prompts import BASIC_RAG_AGENT_INSTRUCTION; print('✓ Import successful')"`

Expected: `✓ Import successful`

**Step 4: Commit**

```bash
git add src/prompts/basic_rag_instruction.py src/prompts/__init__.py
git commit -m "feat: add BasicRAG agent system prompt

Add instruction prompt for BasicRAG agent emphasizing:
- Vector-based document retrieval
- Source citation with relevance scores
- Accuracy over completeness
- Honest uncertainty when insufficient info

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create BasicRAG Agent Core Structure

**Files:**
- Create: `src/agents/agentic_basic_rag.py`

**Step 1: Create agent file with imports and dependencies**

Create `src/agents/agentic_basic_rag.py`:

```python
import asyncio
from loguru import logger
from dataclasses import dataclass

from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.console import Console
from rich.markdown import Markdown

from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.models import RetrievalInfo
from src.settings import settings
from src.prompts import BASIC_RAG_AGENT_INSTRUCTION
from src.retrieval.basic_rag import BasicRAG

logger.remove()
logger.add("agent.log", rotation="10 MB", level="DEBUG")


@dataclass
class BasicRAGDependencies:
    basic_rag: BasicRAG
    top_k: int = 5


def get_openai_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=settings.llm_model,
        provider=OpenAIProvider(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        ),
        settings=ModelSettings(
            temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens
        ),
    )


basic_rag_agent = Agent(
    model=get_openai_model(),
    system_prompt=BASIC_RAG_AGENT_INSTRUCTION,
    deps_type=BasicRAGDependencies,
    retries=2,
)
```

**Step 2: Verify structure compiles**

Run: `uv run python -c "from src.agents.agentic_basic_rag import basic_rag_agent; print('✓ Agent structure loads')"`

Expected: `✓ Agent structure loads`

**Step 3: Commit**

```bash
git add src/agents/agentic_basic_rag.py
git commit -m "feat: add BasicRAG agent core structure

Initialize agent with:
- Dependencies dataclass (BasicRAG + top_k)
- OpenAI model configuration
- System prompt integration
- Retry logic (retries=2)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement BasicRAG Search Tool

**Files:**
- Modify: `src/agents/agentic_basic_rag.py`

**Step 1: Add the search tool**

Add this function after the `basic_rag_agent` declaration in `src/agents/agentic_basic_rag.py`:

```python
@basic_rag_agent.tool
async def search_basic_rag(
    ctx: RunContext[BasicRAGDependencies], query: str
) -> tuple[list[RetrievalInfo], str]:
    """
    Search the vector database and generate an answer.

    Args:
        ctx: Context containing BasicRAG instance and config.
        query: The search query.

    Returns:
        Tuple of (retrieval_infos, generated_answer).
    """
    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    logger.info(f"Tool executing search for: {query}")

    try:
        basic_rag = ctx.deps.basic_rag

        # Call generate with return_context to get both retrieval results and answer
        retrieval_infos, response = basic_rag.generate(
            query, top_k=ctx.deps.top_k, return_context=True
        )

        logger.debug(
            f"Retrieved: {len(retrieval_infos)} documents, response length: {len(response)}"
        )

        # Handle empty results gracefully
        if not retrieval_infos:
            logger.warning(f"No results for query: {query}")
            return [], "No relevant documents found for your query."

        return retrieval_infos, response

    except Exception as e:
        logger.exception(f"BasicRAG search failed: {e}")
        raise ModelRetry(f"Search encountered an error. Try rephrasing your query.")
```

**Step 2: Verify tool registration**

Run: `uv run python -c "from src.agents.agentic_basic_rag import basic_rag_agent; print(f'✓ Tools: {[t.name for t in basic_rag_agent.tools]}')"`

Expected: `✓ Tools: ['search_basic_rag']`

**Step 3: Commit**

```bash
git add src/agents/agentic_basic_rag.py
git commit -m "feat: add search tool to BasicRAG agent

Implement search_basic_rag tool with:
- Input validation (non-empty query)
- BasicRAG.generate() with return_context
- Error handling with ModelRetry
- Empty results handling
- Detailed logging to file

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Interactive CLI Interface

**Files:**
- Modify: `src/agents/agentic_basic_rag.py`

**Step 1: Add async user input helper**

Add this function after the `search_basic_rag` tool in `src/agents/agentic_basic_rag.py`:

```python
async def get_user_input(console: Console) -> str:
    """Runs blocking input in an executor to keep asyncio loop healthy."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, Prompt.ask, "\n[bold green]You[/]")
```

**Step 2: Add main async function**

Add this function at the end of `src/agents/agentic_basic_rag.py`:

```python
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="BasicRAG Knowledge Agent")
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Qdrant collection name to query",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    args = parser.parse_args()

    # Initialize BasicRAG
    basic_rag = BasicRAG(qdrant_collection_name=args.collection_name)
    console = Console()

    messages: list[ModelMessage] = []

    console.print(
        Panel(
            f"[bold blue]BasicRAG Knowledge Agent[/]\n"
            f"Collection: {args.collection_name}\n"
            f"Type 'exit' to quit.",
            expand=False,
        )
    )

    try:
        while True:
            user_input = await get_user_input(console)

            # Exit handling
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[bold red]Goodbye![/]")
                break

            # Input validation
            if not user_input or not user_input.strip():
                console.print("[yellow]Please enter a question.[/yellow]")
                continue

            if len(user_input) > 1000:
                console.print(
                    "[yellow]Query too long. Please keep it under 1000 characters.[/yellow]"
                )
                continue

            deps = BasicRAGDependencies(basic_rag=basic_rag, top_k=args.top_k)

            console.print("\n[bold purple]Assistant[/]")

            # Using a spinner for the initial connection/tool usage phase
            with Live(
                Spinner("dots", text="Thinking..."),
                console=console,
                refresh_per_second=10,
            ) as live:
                try:
                    async with basic_rag_agent.run_stream(
                        user_input, message_history=messages, deps=deps
                    ) as result:
                        accumulated_text = ""
                        async for message in result.stream_text(delta=True):
                            accumulated_text += message
                            # Update Live display with rendered Markdown
                            live.update(Markdown(accumulated_text))

                    # Persist history
                    messages.extend(result.all_messages())

                except ModelRetry as e:
                    # Expected retry from tool - show friendly message
                    live.update(Panel(f"[yellow]{str(e)}[/yellow]", border_style="yellow"))
                except Exception as e:
                    logger.exception(f"Stream error: {e}")
                    live.update(
                        Panel(f"[bold red]Error:[/] {str(e)}", border_style="red")
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    finally:
        logger.info("Session closed.")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Verify the module runs (help text)**

Run: `uv run python -m src.agents.agentic_basic_rag --help`

Expected output showing:
```
usage: agentic_basic_rag.py [-h] --collection_name COLLECTION_NAME [--top_k TOP_K]

BasicRAG Knowledge Agent

options:
  -h, --help            show this help message and exit
  --collection_name COLLECTION_NAME
                        Qdrant collection name to query
  --top_k TOP_K         Number of documents to retrieve
```

**Step 4: Commit**

```bash
git add src/agents/agentic_basic_rag.py
git commit -m "feat: add interactive CLI to BasicRAG agent

Implement async chat loop with:
- Non-blocking user input via executor
- Rich UI with Live markdown streaming
- Input validation (empty, length checks)
- Exit keywords (exit, quit, bye)
- Error display with color-coded panels
- Conversation history persistence
- CLI args (collection_name, top_k)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Refine GraphRAG Agent Error Handling

**Files:**
- Modify: `src/agents/agentic_graph_rag.py`

**Step 1: Enhance search_graphiti tool error handling**

Read the current tool implementation:
```bash
cat src/agents/agentic_graph_rag.py | grep -A 30 "@graphiti_agent.tool"
```

Replace the `search_graphiti` tool in `src/agents/agentic_graph_rag.py` with enhanced version:

```python
@graphiti_agent.tool
async def search_graphiti(
    ctx: RunContext[GraphitiDependencies], query: str
) -> tuple[list[GraphitiNodeInfo], list[GraphitiEdgeInfo], list[GraphitiEpisodeInfo]]:
    """
    Search the Graphiti knowledge graph.

    Args:
        ctx: Context containing retrieval dependencies.
        query: The semantic search query.
    """
    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    graph_retrieval = ctx.deps.graph_retrieval

    logger.info(f"Tool executing search for: {query}")

    try:
        results = await graph_retrieval.retrieve(query, num_results=ctx.deps.top_k)
        node_infos, edge_infos, episode_infos = results

        logger.debug(f"Retrieved: {len(node_infos)} nodes, {len(edge_infos)} edges")

        # Handle empty results gracefully - return empty but valid response
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

**Step 2: Add input validation to main loop**

Find the main loop in `src/agents/agentic_graph_rag.py` and add validation after getting user input.

Locate this section:
```python
user_input = await get_user_input(console)

if user_input.lower() in ["exit", "quit", "bye"]:
    console.print("[bold red]Goodbye![/]")
    break
```

Add validation after the exit check:

```python
user_input = await get_user_input(console)

if user_input.lower() in ["exit", "quit", "bye"]:
    console.print("[bold red]Goodbye![/]")
    break

# Input validation
if not user_input or not user_input.strip():
    console.print("[yellow]Please enter a question.[/yellow]")
    continue

if len(user_input) > 1000:
    console.print(
        "[yellow]Query too long. Please keep it under 1000 characters.[/yellow]"
    )
    continue
```

**Step 3: Enhance stream error handling**

Find the stream handling section in `src/agents/agentic_graph_rag.py`:

```python
try:
    async with graphiti_agent.run_stream(...) as result:
        # streaming code
except Exception as e:
    logger.error(f"Stream error: {e}")
    live.update(Panel(f"[bold red]Error:[/]\\n{str(e)}", border_style="red"))
```

Replace with enhanced error handling:

```python
try:
    async with graphiti_agent.run_stream(
        user_input, message_history=messages, deps=deps
    ) as result:
        accumulated_text = ""
        async for message in result.stream_text(delta=True):
            accumulated_text += message
            # Update Live display with rendered Markdown
            live.update(Markdown(accumulated_text))

    # Persist history
    messages.extend(result.all_messages())

except ModelRetry as e:
    # Expected retry from tool - show friendly message
    live.update(Panel(f"[yellow]{str(e)}[/yellow]", border_style="yellow"))
except Exception as e:
    logger.exception(f"Stream error: {e}")
    live.update(Panel(f"[bold red]Error:[/] {str(e)}", border_style="red"))
```

**Step 4: Verify GraphRAG agent still loads**

Run: `uv run python -c "from src.agents.agentic_graph_rag import graphiti_agent; print('✓ GraphRAG agent loads with refinements')"`

Expected: `✓ GraphRAG agent loads with refinements`

**Step 5: Commit**

```bash
git add src/agents/agentic_graph_rag.py
git commit -m "refactor: enhance GraphRAG agent error handling

Improvements:
- Input validation (empty queries, length limits)
- Specific error types (timeout, connection, generic)
- Empty results return gracefully (no retry)
- ModelRetry exceptions shown as warnings (yellow)
- Detailed exception logging for debugging

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Agent Module Exports

**Files:**
- Create: `src/agents/__init__.py`

**Step 1: Create agents __init__.py**

Create `src/agents/__init__.py`:

```python
from .agentic_basic_rag import basic_rag_agent, BasicRAGDependencies
from .agentic_graph_rag import graphiti_agent, GraphitiDependencies

__all__ = [
    "basic_rag_agent",
    "BasicRAGDependencies",
    "graphiti_agent",
    "GraphitiDependencies",
]
```

**Step 2: Verify imports work**

Run: `uv run python -c "from src.agents import basic_rag_agent, graphiti_agent; print('✓ All agents importable')"`

Expected: `✓ All agents importable`

**Step 3: Commit**

```bash
git add src/agents/__init__.py
git commit -m "feat: add agents module exports

Export both agents and their dependencies for clean imports:
- basic_rag_agent + BasicRAGDependencies
- graphiti_agent + GraphitiDependencies

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Manual Testing & Verification

**Files:**
- None (testing only)

**Step 1: Test BasicRAG agent help text**

Run: `uv run python -m src.agents.agentic_basic_rag --help`

Expected: Help text showing `--collection_name` and `--top_k` options

**Step 2: Verify agent can initialize (dry run)**

Run:
```bash
uv run python -c "
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies
from src.retrieval.basic_rag import BasicRAG

# Test agent can be initialized with dependencies
basic_rag = BasicRAG(qdrant_collection_name='test')
deps = BasicRAGDependencies(basic_rag=basic_rag, top_k=5)
print(f'✓ Agent initialized with deps: {deps}')
"
```

Expected: `✓ Agent initialized with deps: BasicRAGDependencies(basic_rag=<...>, top_k=5)`

**Step 3: Verify GraphRAG agent still works**

Run:
```bash
uv run python -c "
from src.agents.agentic_graph_rag import graphiti_agent
print(f'✓ GraphRAG agent has {len(graphiti_agent.tools)} tool(s)')
"
```

Expected: `✓ GraphRAG agent has 1 tool(s)`

**Step 4: Check logging configuration**

Run:
```bash
uv run python -c "
from src.agents.agentic_basic_rag import logger
from src.agents.agentic_graph_rag import logger as graph_logger
print('✓ Loggers configured (check agent.log after running)')
"
```

Expected: `✓ Loggers configured (check agent.log after running)`

**Step 5: Document testing results**

Create a brief test summary noting what was verified. No commit needed for this step.

---

## Task 8: Update Documentation

**Files:**
- Modify: `docs/plans/2026-02-05-basic-rag-agent-design.md`

**Step 1: Add implementation completion section**

Add this section at the end of `docs/plans/2026-02-05-basic-rag-agent-design.md`:

```markdown
## Implementation Status

**Completed:** 2026-02-05

### Files Created
- `src/prompts/basic_rag_instruction.py` - System prompt for BasicRAG agent
- `src/agents/agentic_basic_rag.py` - Agent implementation with tool and CLI
- `src/agents/__init__.py` - Module exports for both agents

### Files Modified
- `src/prompts/__init__.py` - Added BASIC_RAG_AGENT_INSTRUCTION export
- `src/agents/agentic_graph_rag.py` - Enhanced error handling

### Verification Completed
- ✓ All imports successful
- ✓ Agent tools registered correctly
- ✓ CLI arguments work as expected
- ✓ Error handling improvements in place
- ✓ Logging configured properly

### Usage

**BasicRAG Agent:**
```bash
uv run python -m src.agents.agentic_basic_rag --collection_name <name> --top_k 5
```

**GraphRAG Agent:**
```bash
uv run python -m src.agents.agentic_graph_rag
```
```

**Step 2: Verify documentation is complete**

Read the updated file to ensure it looks correct:
```bash
cat docs/plans/2026-02-05-basic-rag-agent-design.md | tail -40
```

**Step 3: Commit documentation**

```bash
git add docs/plans/2026-02-05-basic-rag-agent-design.md
git commit -m "docs: add implementation status to design doc

Document completion of:
- Created files (prompts, agent, module exports)
- Modified files (prompts init, GraphRAG refinements)
- Verification results
- Usage instructions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary

**Total Tasks:** 8
**Estimated Time:** 40-60 minutes

**Key Deliverables:**
1. ✅ BasicRAG agent system prompt
2. ✅ BasicRAG agent with search tool
3. ✅ Interactive async CLI with streaming
4. ✅ GraphRAG error handling improvements
5. ✅ Module exports for clean imports
6. ✅ Verification testing
7. ✅ Updated documentation

**Testing Strategy:**
- Import verification after each major component
- CLI help text validation
- Dry-run initialization checks
- Tool registration verification

**Integration Notes:**
- Both agents follow identical patterns (pydantic-ai + Rich UI)
- Loguru configured for file-based debugging
- Error handling with ModelRetry for user-friendly messages
- Conversation history persists across turns
- Input validation prevents common errors

**Next Steps After Implementation:**
1. Test with real Qdrant collection
2. Verify agent responses are helpful and accurate
3. Monitor agent.log for any issues
4. Consider adding tests for tool functions
5. Evaluate if shared utilities module is needed
