import asyncio
from dataclasses import dataclass

from rich.live import Live
from rich.markdown import Markdown
from rich.console import Console

from loguru import logger

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from src.models.graphiti_search_info import (
    GraphitiNodeInfo,
    GraphitiEdgeInfo,
    GraphitiEpisodeInfo,
)
from src.settings import settings
from src.prompts import GRAPHITI_AGENT_INSTRUCTION
from src.retrieval.graph_rag import GraphRetrieval


@dataclass
class GraphitiDependencies:
    graph_retrieval: GraphRetrieval
    top_k: int = 5


def get_openai_model():
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
)


@graphiti_agent.tool
async def search_graphiti(
    ctx: RunContext[GraphitiDependencies], query: str
) -> tuple[list[GraphitiNodeInfo], list[GraphitiEdgeInfo], list[GraphitiEpisodeInfo]]:
    """Search the Graphiti knowledge graph with the given query.

    Args:
        ctx: The run context containing dependencies
        query: The search query to find information in the knowledge graph

    Returns:
        A tuple of lists containing node, edge, and episode information that match the query
    """
    graph_retrieval = ctx.deps.graph_retrieval

    try:
        node_infos, edge_infos, episode_infos = await graph_retrieval.retrieve(
            query, num_results=ctx.deps.top_k
        )

        logger.info(f"Node: {len(node_infos)}")
        logger.info(f"Edge: {len(edge_infos)}")
        logger.info(f"Episode: {len(episode_infos)}")

        return node_infos, edge_infos, episode_infos
    except Exception as e:
        logger.error(f"Error searching Graphiti: {str(e)}")
        raise


async def main():
    graph_retrieval = GraphRetrieval()
    top_k = 5

    console = Console()
    messages = []

    try:
        while True:
            user_input = input("\n[You] ")

            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("Goodbye!")
                break

            try:
                print("\n[Assistant]")
                with Live("", console=console, vertical_overflow="visible") as live:
                    deps = GraphitiDependencies(
                        graph_retrieval=graph_retrieval, top_k=top_k
                    )

                    async with graphiti_agent.run_stream(
                        user_input, message_history=messages, deps=deps
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))

                    messages.extend(result.all_messages())

            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        await graph_retrieval.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"\nUnexpected error: {str(e)}")
        raise
