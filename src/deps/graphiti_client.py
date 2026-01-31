from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from loguru import logger
from src.settings import settings
from .openai_client_wrapper import OpenAIClient as OpenAILLMClientWrapper


class GraphitiClient:
    def __init__(self):
        self.llm = OpenAILLMClientWrapper(
            config=LLMConfig(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                small_model=settings.llm_model,
            )
        )

        self.embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=settings.embedding_api_key,
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dimensions,
                base_url=settings.embedding_base_url,
            )
        )

        self.cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(
                api_key=settings.reranker_api_key,
                model=settings.reranker_model,
                base_url=settings.reranker_base_url,
            )
        )

        self.driver = Neo4jDriver(
            uri=settings.graph_db_uri,
            user=settings.graph_db_username,
            password=settings.graph_db_password,
        )

    async def create_client(
        self,
        clear_existing_graphdb_data: bool = False,
        max_coroutines: int = 1,
    ) -> Graphiti:
        logger.info("Creating Graphiti client instance")

        graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=self.llm,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            max_coroutines=max_coroutines,
        )

        # Initialize the graph database with graphiti's indices and constraints
        logger.debug("Building indices and constraints...")
        await graphiti.build_indices_and_constraints()
        logger.info("Graphiti indices and constraints ready")

        # Optionally clear existing data
        if clear_existing_graphdb_data:
            logger.warning("Clearing existing graph data (destructive operation)...")
            await clear_data(graphiti.driver)
            logger.info("Graph data cleared successfully")

        logger.info("Graphiti client created successfully")
        return graphiti

    async def close(self):
        try:
            if self.driver:
                await self.driver.close()
                logger.info("Graph database driver connection closed")
        except Exception as e:
            logger.exception(f"Error closing graph database driver: {e}")
