import asyncio
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from graphiti_core.search.search_config import (
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    NodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    EpisodeReranker,
    SearchConfig,
    SearchResults,
)

from src.settings import settings
from src.deps import OpenAILLMClient, GraphitiClient
from src.models import GraphitiEdgeInfo, GraphitiNodeInfo, GraphitiEpisodeInfo
from src.prompts import RAG_GENERATION_PROMPT

console = Console()


def get_node_edge_episode_infos(
    result: SearchResults,
) -> tuple[list[GraphitiNodeInfo], list[GraphitiEdgeInfo], list[GraphitiEpisodeInfo]]:
    nodes = result.nodes or []
    edges = result.edges or []
    episodes = result.episodes or []

    node_infos = [
        GraphitiNodeInfo(
            uuid=n.uuid,
            summary=n.summary,
        )
        for n in nodes
    ]
    edge_infos = [
        GraphitiEdgeInfo(
            uuid=e.uuid,
            fact=e.fact,
            invalid_at=str(e.invalid_at),
            valid_at=str(e.valid_at),
        )
        for e in edges
    ]
    episode_infos = [
        GraphitiEpisodeInfo(
            uuid=ep.uuid,
            content=ep.content,
        )
        for ep in episodes
    ]

    return node_infos, edge_infos, episode_infos


class GraphRetrieval:
    def __init__(self):
        self.graphiti_client = GraphitiClient()
        self.graphiti = None
        self.llm_client = OpenAILLMClient(
            base_url=settings.llm_base_url,
            api_keys=settings.llm_api_key,
            model_id=settings.llm_model,
        )

    async def initialize_graphiti_client(self):
        if self.graphiti is None:
            self.graphiti = await self.graphiti_client.create_client(
                clear_existing_graphdb_data=False,
                max_coroutines=1,
            )

    async def retrieve(
        self, query: str, num_results: int = 10
    ) -> tuple[
        list[GraphitiNodeInfo], list[GraphitiEdgeInfo], list[GraphitiEpisodeInfo]
    ]:
        await self.initialize_graphiti_client()

        with console.status("[bold green]Searching graph database...", spinner="dots"):
            config = SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[
                        EdgeSearchMethod.bm25,
                        EdgeSearchMethod.cosine_similarity,
                        EdgeSearchMethod.bfs,
                    ],
                    reranker=EdgeReranker.rrf,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[
                        NodeSearchMethod.bm25,
                        NodeSearchMethod.cosine_similarity,
                        NodeSearchMethod.bfs,
                    ],
                    reranker=NodeReranker.rrf,
                ),
                episode_config=EpisodeSearchConfig(
                    search_methods=[
                        EpisodeSearchMethod.bm25,
                    ],
                    reranker=EpisodeReranker.rrf,
                ),
                limit=num_results,
            )
            results = await self.graphiti._search(query, config)

            node_infos, edge_infos, episode_infos = get_node_edge_episode_infos(results)

            logger.info(f"Node: {len(node_infos)}")
            logger.info(f"Edge: {len(edge_infos)}")
            logger.info(f"Episode: {len(episode_infos)}")

        return node_infos, edge_infos, episode_infos

    async def generate(
        self, query: str, num_results: int = 10
    ) -> tuple[list[dict], str]:
        node_infos, edge_infos, episode_infos = await self.retrieve(
            query, num_results=num_results
        )

        contexts_data = []
        if len(node_infos) > 0:
            node_content = "Node Content:"
            for n in node_infos:
                node_content += f"\n- {n.summary}"
            contexts_data.append(node_content)

        if len(edge_infos) > 0:
            edge_content = "Edge Content:"
            for e in edge_infos:
                edge_content += (
                    f"\n- {e.fact} (Valid from {e.valid_at} to {e.invalid_at})"
                )
            contexts_data.append(edge_content)

        if len(episode_infos) > 0:
            episode_content = "Episode Content:"
            for ep in episode_infos:
                episode_content += f"\n- {ep.content}"
            contexts_data.append(episode_content)

        context_block = "\n---\n".join(contexts_data)

        prompt = RAG_GENERATION_PROMPT.format(context_block=context_block, query=query)

        with console.status("[bold blue]Generating answer...", spinner="bouncingBar"):
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        return contexts_data, response

    async def close(self):
        """Close the graph database connection"""
        if self.graphiti_client:
            await self.graphiti_client.close()


def display_results(query: str, contexts: list[str], response: str):
    # 1. Show the Query
    console.print(f"\n[bold magenta]Query:[/bold magenta] [italic]{query}[/italic]\n")

    # 2. Show Results Table
    table = Table(
        title="Retrieved Facts from Knowledge Graph", show_lines=True, expand=True
    )
    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Content", style="white")

    for i, context in enumerate(contexts):
        table.add_row(f"Context {i + 1}", context)

    console.print(table)

    # 3. Show Final Answer in a Panel
    # Using Markdown inside the panel so your bolding/bullets render correctly
    md_answer = Markdown(response)
    console.print(
        Panel(
            md_answer,
            title="[bold green]LLM Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


async def main_async():
    retrieval = GraphRetrieval()
    console.print(Panel.fit("Graph RAG Evaluation CLI Mode", style="bold cyan"))

    try:
        while True:
            try:
                query = console.input(
                    "[bold yellow]Enter a question (or 'exit'): [/bold yellow]"
                )
                if query.lower() in ["exit", "quit"]:
                    break

                contexts, response = await retrieval.generate(query, num_results=5)
                display_results(query, contexts, response)
            except Exception as e:
                console.print(f"[bold red]Error: {e}[/bold red]")
    finally:
        await retrieval.close()


if __name__ == "__main__":
    asyncio.run(main_async())
