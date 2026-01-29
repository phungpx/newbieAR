from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

from src.settings import settings
from src.deps import LLMClient, OpenAIEmbedding, QdrantVectorStore

# Initialize Rich Console
console = Console()


class Retrieval:
    def __init__(self):
        self.vector_store = QdrantVectorStore(
            uri=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.embedding_client = OpenAIEmbedding(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model_id=settings.embedding_model_id,
        )
        self.llm_client = LLMClient(
            base_url=settings.llm_base_url,
            api_keys=settings.llm_api_key,
            model_id=settings.llm_model,
        )

    def generate(self, query: str, limit: int = 5) -> tuple[list[str], str]:
        # 1. Embedding & Retrieval
        with console.status("[bold green]Searching knowledge base...", spinner="dots"):
            embedding = self.embedding_client.embed([query])
            retrieved_documents = self.vector_store.query(
                collection_name=settings.qdrant_collection_name,
                query_vector=embedding[0],
                top_k=limit,
            )

        # Store as dictionaries for easier Rich Table rendering
        contexts_data = []
        raw_contexts_for_prompt = []

        for doc in retrieved_documents.points:
            content = doc.payload.get("content", "")
            source = doc.payload.get("sources", "Unknown")
            score = getattr(doc, "score", 0.0)  # Qdrant usually returns a score

            contexts_data.append({"content": content, "source": source, "score": score})
            raw_contexts_for_prompt.append(f"document: {content}, source: {source}")

        # 2. Prompt Construction
        prompt_start = """
        # ROLE
        You are a precise Technical Support Assistant. Your goal is to answer questions based strictly on the provided documentation context.

        # RULES OF ENGAGEMENT
        1. **Greeting Logic:** If the user provides a general greeting (e.g., "Hi", "Hello"), respond with a friendly greeting and do not reference the documentation.
        2. **Contextual Fidelity:** Only answer using the provided Context. If the answer is not contained within the Context, respond exactly with: "I don't know."
        3. **Citation Requirement:** For every specific claim or instruction you provide, you MUST cite the source. Use the format: [Source Name, Page X].
        4. **Tone:** Maintain a professional, helpful, and concise tone. Avoid fluff or repetitive introductory phrases.

        # RESPONSE FORMAT
        - Use bullet points for steps or lists.
        - Bold key terms for readability.
        - Place citations at the end of the relevant sentence or paragraph.

        # CONTEXT:
        """

        context_block = "\n---\n".join(raw_contexts_for_prompt)
        prompt_end = f"---\n# USER QUESTION: \n{query}\n\n# ANSWER:"

        # 3. LLM Generation
        with console.status("[bold blue]Generating answer...", spinner="bouncingBar"):
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt_start},
                    {"role": "user", "content": context_block},
                    {"role": "user", "content": prompt_end},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        return contexts_data, response


def display_results(query: str, contexts: list[dict], response: str):
    # 1. Show the Query
    console.print(f"\n[bold magenta]Query:[/bold magenta] [italic]{query}[/italic]\n")

    # 2. Show Retrieval Table
    table = Table(title="Retrieved Documents", show_lines=True, expand=True)
    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Content (Preview)", style="white")

    for i, ctx in enumerate(contexts):
        # Preview first 150 chars of content
        preview = ctx["content"].replace("\n", " ") + "..."
        table.add_row(str(i + 1), f"{ctx['score']:.4f}", ctx["source"], preview)

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


if __name__ == "__main__":
    retrieval = Retrieval()
    console.print(Panel.fit("RAG Evaluation CLI Mode", style="bold cyan"))

    while True:
        try:
            query = console.input(
                "[bold yellow]Enter a question (or 'exit'): [/bold yellow]"
            )
            if query.lower() in ["exit", "quit"]:
                break

            contexts, response = retrieval.generate(query, limit=5)
            display_results(query, contexts, response)

        except KeyboardInterrupt:
            break
