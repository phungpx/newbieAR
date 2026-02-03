from pathlib import Path
from loguru import logger
from deepeval.models.llms import GPTModel
from deepeval.models.embedding_models import LocalEmbeddingModel
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
)
from src.settings import settings

# from src.evals.bedrock_llm_wrapper import BedrockLLMWrapper
from deepeval.synthesizer import Synthesizer, Evolution
from .utils import save_goldens_to_files
from .config import STYLING_CONFIG

TOPIC = "wikipedia_article"

model = GPTModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    cost_per_input_token=0.3 * 10**-6,
    cost_per_output_token=2.5 * 10**-6,
)

# model = BedrockLLMWrapper(
#     model=settings.critique_model_name,
#     region_name=settings.critique_model_region_name,
# )


embeder = LocalEmbeddingModel(
    model=settings.embedding_model,
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key,
)

# Apply for filtering the generated query by the critic model
filtration_config = FiltrationConfig(
    # Relax threshold to avoid over-filtering all generated samples
    synthetic_input_quality_threshold=0.3,
    max_quality_retries=1,
    critic_model=model,
)

# Apply for evolving query by combining the original query and retrieved context
evolution_config = EvolutionConfig(
    evolutions={
        Evolution.MULTICONTEXT: 1 / 7,
        Evolution.CONCRETIZING: 1 / 7,
        Evolution.CONSTRAINED: 1 / 7,
        Evolution.COMPARATIVE: 1 / 7,
        Evolution.HYPOTHETICAL: 1 / 7,
        Evolution.IN_BREADTH: 1 / 7,
        Evolution.REASONING: 1 / 7,
    },
    num_evolutions=2,
)

# Apply for generating the first query from retrieved context
styling_config = StylingConfig(
    input_format=STYLING_CONFIG[TOPIC]["input_format"],
    expected_output_format=STYLING_CONFIG[TOPIC]["expected_output_format"],
    task=STYLING_CONFIG[TOPIC]["task"],
    scenario=STYLING_CONFIG[TOPIC]["scenario"],
)

# Settings for building RAG
context_construction_config = ContextConstructionConfig(
    embedder=embeder,
    critic_model=model,
    encoding="utf-8",
    chunk_size=1024,
    chunk_overlap=20,
    max_contexts_per_document=5,
    min_contexts_per_document=3,
    max_context_length=5,
    min_context_length=3,
)

synthesizer = Synthesizer(
    model=model,
    async_mode=False,
    # max_concurrent=100,
    filtration_config=filtration_config,
    evolution_config=evolution_config,
    styling_config=styling_config,
    cost_tracking=True,
)


file_dir = Path("data/wikipedia/files")
file_paths = list(file_dir.glob("**/*.*"))
logger.info(f"Found {len(file_paths)} files in {file_dir}")

output_dir = Path("data/goldens")
output_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Saving goldens to {output_dir}")

for file_path in file_paths:
    logger.info(f"Synthesizing {file_path}")
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(file_path)],
        include_expected_output=True,
        context_construction_config=context_construction_config,
        max_goldens_per_context=3,
    )
    logger.info(f"Synthesis cost: {synthesizer.synthesis_cost}")
    save_goldens_to_files(goldens, output_dir)


# # Save as JSON with a custom filename my_dataset.json
# synthesizer.save_as(
#     file_type="json",
#     directory="RAG/data/goldens",
#     file_name=Path(file_path).stem,
# )
