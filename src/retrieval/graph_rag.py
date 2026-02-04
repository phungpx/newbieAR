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

# Initialize Rich Console
console = Console()


def get_node_edge_episode_contents(
    result: SearchResults,
) -> tuple[list[str], list[str], list[str]]:
    nodes = result.nodes or []
    edges = result.edges or []
    episodes = result.episodes or []

    node_contents = [n.summary for n in nodes if n.summary]
    edge_contents = [e.fact for e in edges if e.fact]
    episode_contents = [ep.content for ep in episodes if ep.content]

    return node_contents, edge_contents, episode_contents


class GraphRetrieval:
    def __init__(self):
        self.graphiti_client = GraphitiClient()
        self.graphiti = None
        self.llm_client = OpenAILLMClient(
            base_url=settings.llm_base_url,
            api_keys=settings.llm_api_key,
            model_id=settings.llm_model,
        )

    async def initialize(self):
        if self.graphiti is None:
            self.graphiti = await self.graphiti_client.create_client(
                clear_existing_graphdb_data=False,
                max_coroutines=1,
            )

    async def retrieve(self, query: str, num_results: int = 10):
        if self.graphiti is None:
            await self.initialize()

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

            node_contents, edge_contents, episode_contents = (
                get_node_edge_episode_contents(results)
            )

            logger.info(f"Node: {len(results.nodes)}")
            logger.info(f"Edge: {len(results.edges)}")
            logger.info(f"Episode: {len(results.episodes)}")

        return node_contents, edge_contents, episode_contents

    async def generate(
        self, query: str, num_results: int = 10
    ) -> tuple[list[dict], str]:
        """Retrieve results and generate an answer using LLM"""
        # 1. Retrieve from graph database
        node_contents, edge_contents, episode_contents = await self.retrieve(
            query, num_results=num_results
        )

        contexts_data = []
        if len(node_contents) > 0:
            node_content = "Node Content:"
            for i, text in enumerate(node_contents):
                node_content += f"\n- Node {i + 1}: {text}"
            contexts_data.append(node_content)

        if len(edge_contents) > 0:
            edge_content = "Edge Content:"
            for i, text in enumerate(edge_contents):
                edge_content += f"\n- Edge {i + 1}: {text}"
            contexts_data.append(edge_content)

        if len(episode_contents) > 0:
            episode_content = "Episode Content:"
            for i, text in enumerate(episode_contents):
                episode_content += f"\n- Episode {i + 1}: {text}"
            contexts_data.append(episode_content)

        context_block = "\n---\n".join(contexts_data)

        # 2. Prompt Construction
        prompt_start = """
# ROLE
You are a precise Knowledge Assistant. Your goal is to answer questions based strictly on the provided facts from the knowledge graph.

# RULES OF ENGAGEMENT
1. **Greeting Logic:** If the user provides a general greeting (e.g., "Hi", "Hello"), respond with a friendly greeting and do not reference the facts.
2. **Contextual Fidelity:** Only answer using the provided Facts. If the answer is not contained within the Facts, respond exactly with: "I don't know."
3. **Fact Integration:** Synthesize information from multiple facts when relevant to provide a comprehensive answer.
4. **Temporal Awareness:** Pay attention to validity periods if mentioned in the facts.
5. **Tone:** Maintain a professional, helpful, and concise tone. Avoid fluff or repetitive introductory phrases.

# RESPONSE FORMAT
- Use bullet points for steps or lists.
- Bold key terms for readability.
- Synthesize information from multiple facts when relevant.

# FACTS FROM KNOWLEDGE GRAPH:
"""

        prompt_end = f"---\n# USER QUESTION: \n{query}\n\n# ANSWER:"

        prompt = prompt_start + context_block + prompt_end

        # 3. LLM Generation
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
    """Display search results and LLM response in formatted tables"""
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
