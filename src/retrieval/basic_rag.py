from src.settings import settings
from src.models import RetrievalInfo
from src.prompts import RAG_GENERATION_PROMPT
from src.deps import OpenAIEmbedding, QdrantVectorStore, OpenAILLMClient
from src.retrieval.utils import display_rag_results


class BasicRAG:
    def __init__(self, qdrant_collection_name: str = None):
        self.collection_name = qdrant_collection_name or settings.qdrant_collection_name
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

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RetrievalInfo]:
        import asyncio

        embedding = await asyncio.to_thread(self.embedder.embed_texts, [query])
        retrieved_documents = await asyncio.to_thread(
            self.vector_store.query,
            collection_name=self.collection_name,
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

        if score_threshold > 0.0:
            retrieval_infos = [r for r in retrieval_infos if r.score >= score_threshold]

        if self.cross_encoder is not None and retrieval_infos:
            passages = [r.content for r in retrieval_infos]
            ranked = await self.cross_encoder.rank(query, passages)
            score_map = {passage: score for passage, score in ranked}
            for r in retrieval_infos:
                r.score = score_map.get(r.content, r.score)
            retrieval_infos.sort(key=lambda r: r.score, reverse=True)

        return retrieval_infos

    async def generate(
        self,
        query: str,
        top_k: int = 5,
        return_context: bool = False,
    ) -> tuple[list[RetrievalInfo], str] | str:
        retrieval_infos = await self.retrieve(query, top_k)

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


if __name__ == "__main__":
    import asyncio
    import argparse
    from rich.panel import Panel
    from rich.console import Console

    async def _main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--qdrant_collection_name", type=str, required=True)
        parser.add_argument("--top_k", type=int, default=10)
        args = parser.parse_args()

        basic_rag = BasicRAG(qdrant_collection_name=args.qdrant_collection_name)
        console = Console()
        console.print(
            Panel.fit(
                f"Basic Retrieval CLI Mode - Collection: {args.qdrant_collection_name}",
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
                retrieval_infos, response = await basic_rag.generate(
                    query, top_k=args.top_k, return_context=True
                )
                contexts = [r.content for r in retrieval_infos]
                citations = [r.source for r in retrieval_infos]
                display_rag_results(console, query, contexts, citations, response)
            except KeyboardInterrupt:
                break

    asyncio.run(_main())
