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
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider

from src.models import RetrievalInfo
from src.settings import settings
from src.prompts import AGENTIC_RAG_INSTRUCTION
from src.retrieval.basic_rag import BasicRAG

logger.remove()
logger.add(sink="logs/agentic_basic_rag.log", rotation="10 MB", level="DEBUG")


@dataclass
class BasicRAGDependencies:
    basic_rag: BasicRAG
    top_k: int = 5
    citations: list | None = None


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


def get_google_model() -> GoogleModel:
    model_name = "gemini-2.5-flash"
    project_id = "vns-durian-traceability"
    return GoogleModel(
        model_name=model_name,
        provider=GoogleProvider(project=project_id, vertexai=True),
        settings=ModelSettings(
            temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens
        ),
    )


basic_rag_agent = Agent(
    system_prompt=AGENTIC_RAG_INSTRUCTION,
    deps_type=BasicRAGDependencies,
    retries=2,
)


@basic_rag_agent.tool
async def search_basic_rag(
    ctx: RunContext[BasicRAGDependencies], query: str
) -> tuple[list[RetrievalInfo], str]:
    """
    Search the vector database and generate an answer.

    Args:
        ctx: Context containing BasicRAG instance and config
        query: The search query

    Returns:
        Tuple of (retrieval_infos, generated_answer)
    """
    basic_rag = ctx.deps.basic_rag

    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    # Log to file, not console, to keep UI clean
    logger.info(f"Tool executing search for: {query}")

    try:
        retrieval_infos, generated_answer = await asyncio.to_thread(
            basic_rag.generate, query, top_k=ctx.deps.top_k, return_context=True
        )

        logger.debug(
            f"Retrieved {len(retrieval_infos)} documents, answer length: {len(generated_answer)}"
        )

        # Handle empty results gracefully
        if not retrieval_infos:
            logger.warning(f"No results for query: {query}")
            return [], "No relevant documents found in the database."

        return retrieval_infos, generated_answer

    except asyncio.TimeoutError:
        logger.error("Search timed out")
        raise ModelRetry("Search timed out. Try a simpler query.")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise ModelRetry("Database connection failed. Please try again.")
    except Exception as e:
        logger.exception(f"BasicRAG search failed: {e}")
        raise ModelRetry(f"Search encountered an error. Try rephrasing your query.")


async def get_user_input(console: Console) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, Prompt.ask, "\n[bold green]You[/]")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="BasicRAG Knowledge Agent")
    parser.add_argument(
        "--collection_name", type=str, required=True, help="Qdrant collection name"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    args = parser.parse_args()

    # Initialize components
    model = get_openai_model()
    basic_rag = BasicRAG(qdrant_collection_name=args.collection_name)
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

            # Stream response with Live markdown display
            with Live(
                Spinner("dots", text="Thinking..."),
                console=console,
                refresh_per_second=10,
            ) as live:
                try:
                    async with basic_rag_agent.run_stream(
                        user_input, model=model, message_history=messages, deps=deps
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
        logger.info("Session closed.")


if __name__ == "__main__":
    asyncio.run(main())
