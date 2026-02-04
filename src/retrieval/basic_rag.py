from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.settings import settings
from src.models import RetrievalInfo
from src.prompts import RAG_GENERATION_PROMPT
from src.deps import OpenAIEmbedding, QdrantVectorStore, OpenAILLMClient

console = Console()


class BasicRAG:
    def __init__(self):
        self.vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
        self.embedder = OpenAIEmbedding(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model_id=settings.embedding_model,
        )
        self.llm = OpenAILLMClient(
            base_url=settings.llm_base_url,
            api_keys=settings.llm_api_key,
            model_id=settings.llm_model,
        )
        self.cross_encoder = None

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
    ) -> list[RetrievalInfo]:
        with console.status(
            f"[bold green]Searching knowledge base {collection_name}...", spinner="dots"
        ):
            embedding = self.embedder.embed_texts([query])
            retrieved_documents = self.vector_store.query(
                collection_name=collection_name,
                query_vector=embedding[0],
                top_k=top_k,
            )

        retrieval_infos: list[RetrievalInfo] = []

        for doc in retrieved_documents.points:
            content = doc.payload.get("text", "")
            score = getattr(doc, "score", 0.0)
            filename = doc.payload.get("filename", "Unknown")
            chunk_id = doc.payload.get("chunk_id", "Unknown")
            source = f"{filename} - Chunk #{chunk_id}"

            retrieval_infos.append(
                RetrievalInfo(content=content, source=source, score=score)
            )

        return retrieval_infos

    def generate(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        return_context: bool = False,
    ) -> tuple[list[RetrievalInfo], str] | str:
        retrieval_infos = self.retrieve(query, collection_name, top_k)

        context = ""
        for retrieval_info in retrieval_infos:
            context += f"Document: {retrieval_info.content} (Score: {retrieval_info.score:.4f}), Source: {retrieval_info.source}\n"

        prompt = RAG_GENERATION_PROMPT.format(context_block=context, query=query)

        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        if return_context:
            return retrieval_infos, response

        return response


def display_results(
    query: str, retrieval_infos: list[RetrievalInfo], response: str = None
):
    console.print(
        Panel.fit(
            Markdown(query),
            title="[bold green]Question[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    table = Table(title="Retrieved Documents", show_lines=True, expand=True)
    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Content (Preview)", style="white")

    for i, retrieval_info in enumerate(retrieval_infos):
        preview = retrieval_info.content.replace("\n", " ") + "..."
        table.add_row(
            str(i + 1), f"{retrieval_info.score:.4f}", retrieval_info.source, preview
        )

    console.print(table)

    if response is not None:
        console.print(
            Panel.fit(
                Markdown(response),
                title="[bold green]Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    basic_rag = BasicRAG()

    console.print(
        Panel.fit(
            f"Basic Retrieval CLI Mode - Collection: {args.collection_name}",
            style="bold cyan",
        )
    )

    while True:
        try:
            query = console.input(
                "[bold yellow]Enter a question (or 'exit'): [/bold yellow]"
            )
            if query.lower() in ["exit", "quit"]:
                break

            retrieval_infos, response = basic_rag.generate(
                query, args.collection_name, args.top_k, return_context=True
            )
            display_results(query, retrieval_infos, response)

        except KeyboardInterrupt:
            break
