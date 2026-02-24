import asyncio
from loguru import logger

from pydantic_ai import RunContext, ModelRetry

from src.agents.deps import AgentDependencies
from src.agents.agentic_rag import agentic_rag


@agentic_rag.tool
async def search_basic_rag(ctx: RunContext[AgentDependencies], query: str) -> list[str]:
    """
    Search the vector database and generate an answer.

    Args:
        ctx: Context containing retrieval dependencies.
        query: The search query

    Returns:
        contexts: list of contexts
    """
    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    # RAG validation
    if not ctx.deps.basic_rag:
        raise ModelRetry("Rag does not exist.")

    # Log to file, not console, to keep UI clean
    logger.info(f"Tool executing search for: {query}")

    try:
        retrieval_infos = await asyncio.to_thread(
            ctx.deps.basic_rag.retrieve, query, top_k=ctx.deps.top_k
        )

        logger.debug(f"Retrieved {len(retrieval_infos)} documents.")

        contexts, citations = [], []
        for retrieval_info in retrieval_infos:
            contexts.append(
                f"Document: {retrieval_info.content} (Score: {retrieval_info.score:.4f}), Source: {retrieval_info.source}"
            )
            citations.append(retrieval_info.source)

        # Update dependencies
        ctx.deps.contexts = contexts
        ctx.deps.citations = citations

        # Handle empty results gracefully
        if not retrieval_infos:
            logger.warning(f"No results for query: {query}")
            return []

        return contexts

    except asyncio.TimeoutError:
        logger.error("Search timed out")
        raise ModelRetry("Search timed out. Try a simpler query.")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise ModelRetry("Database connection failed. Please try again.")
    except Exception as e:
        logger.exception(f"BasicRAG search failed: {e}")
        raise ModelRetry("Search encountered an error. Try rephrasing your query.")


@agentic_rag.tool
async def search_graphiti(ctx: RunContext[AgentDependencies], query: str) -> list[str]:
    """
    Search the Graphiti knowledge graph and generate an answer.

    Args:
        ctx: Context containing retrieval dependencies.
        query: The semantic search query.

    Returns:
        contexts: list of contexts
    """
    # Input validation
    if not query or not query.strip():
        raise ModelRetry("Query cannot be empty. Please provide a search query.")

    # RAG validation
    if not ctx.deps.graph_rag:
        raise ModelRetry("Graph Rag does not exist.")

    # Log to file, not console, to keep UI clean
    logger.info(f"Tool executing search for: {query}")

    try:
        contexts, citations = await asyncio.to_thread(
            ctx.deps.graph_rag.generate, query, num_results=ctx.deps.top_k
        )

        logger.debug(
            f"Retrieved {len(contexts)} context blocks with {len(citations)} citations"
        )

        # Update dependencies
        ctx.deps.citations = citations
        ctx.deps.contexts = contexts

        # Handle empty results gracefully
        if not contexts:
            logger.warning(f"No results for query: {query}")
            return []

        return contexts

    except asyncio.TimeoutError:
        logger.error("Search timed out")
        raise ModelRetry("Search timed out. Try a simpler query.")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise ModelRetry("Database connection failed. Please try again.")
    except Exception as e:
        logger.exception(f"Graphiti search failed: {e}")
        raise ModelRetry("Search encountered an error. Try rephrasing your query.")
