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

from src.settings import settings
from src.prompts import GRAPHITI_AGENT_INSTRUCTION
from src.retrieval.graph_rag import GraphRetrieval

logger.remove()
logger.add(sink="logs/agentic_graph_rag.log", rotation="10 MB", level="DEBUG")


@dataclass
class GraphitiDependencies:
    graph_retrieval: GraphRetrieval
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


graphiti_agent = Agent(
    model=get_openai_model(),
    system_prompt=GRAPHITI_AGENT_INSTRUCTION,
    deps_type=GraphitiDependencies,
    retries=2,
)


@graphiti_agent.tool
async def search_graphiti(
    ctx: RunContext[GraphitiDependencies], query: str
) -> tuple[list[str], list[str], str]:
    """
    Search the Graphiti knowledge graph and generate an answer.

    Args:
        ctx: Context containing retrieval dependencies.
        query: The semantic search query.

    Returns:
        Tuple of (contexts_data, citations, generated_answer)
    """
    graph_retrieval = ctx.deps.graph_retrieval

    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    # Log to file, not console, to keep UI clean
    logger.info(f"Tool executing search for: {query}")

    try:
        contexts_data, citations, generated_answer = await graph_retrieval.generate(
            query, num_results=ctx.deps.top_k
        )

        logger.debug(
            f"Retrieved {len(contexts_data)} context blocks with {len(citations)} citations"
        )

        # Handle empty results gracefully
        if not contexts_data:
            logger.warning(f"No results for query: {query}")
            return [], [], "No relevant information found in the knowledge graph."

        return contexts_data, citations, generated_answer

    except asyncio.TimeoutError:
        logger.error("Search timed out")
        raise ModelRetry("Search timed out. Try a simpler query.")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise ModelRetry("Database connection failed. Please try again.")
    except Exception as e:
        logger.exception(f"Graphiti search failed: {e}")
        raise ModelRetry(f"Search encountered an error. Try rephrasing your query.")


async def get_user_input(console: Console) -> str:
    """Runs blocking input in an executor to keep asyncio loop healthy."""
    loop = asyncio.get_running_loop()
    # Use Rich's Prompt, but run it in a thread so we don't block background async tasks
    return await loop.run_in_executor(None, Prompt.ask, "\n[bold green]You[/]")


async def main():
    graph_retrieval = GraphRetrieval()
    console = Console()

    messages: list[ModelMessage] = []

    console.print(
        Panel(
            "[bold blue]Graphiti Knowledge Agent[/]\nType 'exit' to quit.", expand=False
        )
    )

    try:
        while True:
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

            deps = GraphitiDependencies(graph_retrieval=graph_retrieval, top_k=5)

            console.print("\n[bold purple]Assistant[/]")

            # Using a spinner for the initial connection/tool usage phase
            with Live(
                Spinner("dots", text="Thinking..."),
                console=console,
                refresh_per_second=10,
            ) as live:
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
                    # Expected retry - show friendly message
                    logger.warning(f"Model retry: {e}")
                    live.update(
                        Panel(f"[yellow]{str(e)}[/yellow]", border_style="yellow")
                    )
                except Exception as e:
                    # Unexpected error
                    logger.exception(f"Stream error: {e}")
                    live.update(
                        Panel(f"[bold red]Error:[/]\n{str(e)}", border_style="red")
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/]")
    finally:
        await graph_retrieval.close()
        logger.info("Session closed.")


if __name__ == "__main__":
    asyncio.run(main())
