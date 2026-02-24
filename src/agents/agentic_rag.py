import asyncio
from loguru import logger

from pydantic_ai.messages import ModelMessage
from pydantic_ai import Agent, ModelRetry

from src.prompts import AGENTIC_RAG_INSTRUCTION
from src.retrieval import BasicRAG, GraphRAG
from src.agents import AgentDependencies
from src.agents.models import get_openai_model, get_google_vertex_model
from src.agents.tools import search_basic_rag, search_graphiti
from src.retrieval.utils import display_rag_results
from src.settings import settings


logger.remove()
logger.add(sink="logs/agentic_rag.log", rotation="10 MB", level="DEBUG")


agentic_rag = Agent(
    system_prompt=AGENTIC_RAG_INSTRUCTION,
    deps_type=AgentDependencies,
    retries=2,
    tools=[search_basic_rag, search_graphiti],
)


async def main():
    import argparse
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.spinner import Spinner
    from rich.console import Console
    from rich.markdown import Markdown

    parser = argparse.ArgumentParser(description="BasicRAG Knowledge Agent")
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    args = parser.parse_args()

    async def get_user_input() -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, Prompt.ask, "\n[bold green]You[/]")

    # Initialize components
    model = get_openai_model(
        model_name=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )

    # Initialize agent dependency
    deps = AgentDependencies(
        basic_rag=BasicRAG(qdrant_collection_name=args.collection_name),
        graph_rag=GraphRAG(),
        top_k=args.top_k,
        citations=None,
        contexts=None,
    )

    console = Console()
    messages: list[ModelMessage] = []

    # Welcome panel
    console.print(
        Panel(
            f"[bold blue]BasicRAG Knowledge Agent[/]\nCollection: {args.collection_name}\nType 'exit' to quit.",
            expand=False,
        )
    )

    try:
        while True:
            # Non-blocking user input
            user_input = await get_user_input()
            logger.info(f"User: {user_input}")

            # Exit handling
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[bold red]Goodbye![/]")
                break

            # Input validation
            if not user_input or not user_input.strip():
                console.print("[yellow]Please enter a question.[/yellow]")
                continue

            if len(user_input) > 1000:
                # TODO: check for the context window size of embedding model if this question is routed to searching tool
                console.print(
                    "[yellow]Query too long. Please keep it under 1000 characters.[/yellow]"
                )
                continue

            console.print("\n[bold purple]Assistant[/]")

            deps.clear_context()

            # Stream response with Live markdown display
            with Live(
                Spinner("dots", text="Thinking..."),
                console=console,
                refresh_per_second=10,
            ) as live:
                try:
                    async with agentic_rag.run_stream(
                        user_input, model=model, message_history=messages, deps=deps
                    ) as result:
                        accumulated_text = ""
                        async for message in result.stream_text(delta=True):
                            accumulated_text += message
                            # Update Live display with rendered Markdown
                            live.update(Markdown(accumulated_text))

                    logger.info(f"Assistant: {accumulated_text}")
                    # Persist history
                    messages.extend(result.all_messages())
                    # Display Context & Citation
                    display_rag_results(
                        console, contexts=deps.contexts, citations=deps.citations
                    )

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
        logger.info("Session closed.")


if __name__ == "__main__":
    asyncio.run(main())
