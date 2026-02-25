from enum import Enum
from pathlib import Path
from loguru import logger

# from src.synthesis.bedrock_model import AmazonBedrockModel
from deepeval.models.llms import GPTModel
from deepeval.models.embedding_models import LocalEmbeddingModel
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
)
from dataclasses import dataclass
from deepeval.synthesizer import Synthesizer, Evolution

from src.settings import settings

# from src.evals.bedrock_llm_wrapper import BedrockLLMWrapper
from src.synthesis.utils import save_goldens_to_files


class Topic(Enum):
    RESEARCH_PAPER = "paper"
    WIKIPEDIA_ARTICLE = "article"


@dataclass
class StylingProfile:
    input_format: str
    expected_output_format: str
    task: str
    scenario: str


STYLING_CONFIG = {
    Topic.RESEARCH_PAPER.value: StylingProfile(
        input_format="Natural language questions about research papers, technical concepts, methodologies, findings, and implementations",
        expected_output_format="Detailed technical responses that include: key concepts and definitions, methodology explanations, experimental findings and results, technical comparisons, implementation details, limitations and future work, and citations to relevant sections. Responses should be precise, technically accurate, and reference specific content from the research papers. Include both high-level summaries and technical depth as appropriate.",
        task="Knowledge retrieval and technical analysis of research papers, including understanding methodologies, experimental results, technical architectures, comparisons with related work, and practical applications",
        scenario="Researchers, engineers, and technical professionals seeking information about research papers, including: understanding proposed methods and architectures, experimental setups and results, technical comparisons with other approaches, implementation details and code examples, limitations and future research directions, and practical applications. Questions may range from high-level overviews to deep technical details. Responses should be accurate, well-structured, and grounded in the paper content.",
    ),
    Topic.WIKIPEDIA_ARTICLE.value: StylingProfile(
        input_format="Natural language questions",
        expected_output_format="Detailed paragraph responses and short structured bios. Include key facts (roles, affiliations, notable works, awards), a concise timeline of major events, and a short bullet list of notable contributions. Provide plain-language explanations for general audiences and optional technical depth when requested.",
        task="Knowledge retrieval and concise biographical summaries for individuals",
        scenario="Users asking about people such as entrepreneurs, scientists, researchers, engineers, and other professionals seeking background information, achievements, affiliations, research contributions, patents, publications, company roles, and notable awards. Responses should be factual, neutral in tone, organized for quick scanning, and reference retrieved context where available.",
    ),
}

TOPIC = Topic.RESEARCH_PAPER.value

model = GPTModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    cost_per_input_token=0.3 * 10**-6,
    cost_per_output_token=2.5 * 10**-6,
)

# import os

# aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
# aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
# aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

# model = AmazonBedrockModel(
#     model=settings.critique_model_name,
#     region=settings.critique_model_region_name,
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key,
#     aws_session_token=aws_session_token,
#     cost_per_input_token=0.3 * 10**-6,
#     cost_per_output_token=2.5 * 10**-6,
#     generation_kwargs={"max_tokens": 7000, "temperature": 0.1, "top_p": 0.9},
# )


embeder = LocalEmbeddingModel(
    model=settings.embedding_model,
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key,
)

# Apply for filtering the generated query by the critic model
filtration_config = FiltrationConfig(
    # Relax threshold to avoid over-filtering all generated samples
    synthetic_input_quality_threshold=0.1,
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
    input_format=STYLING_CONFIG[TOPIC].input_format,
    expected_output_format=STYLING_CONFIG[TOPIC].expected_output_format,
    task=STYLING_CONFIG[TOPIC].task,
    scenario=STYLING_CONFIG[TOPIC].scenario,
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
    max_concurrent=100,
    filtration_config=filtration_config,
    evolution_config=evolution_config,
    styling_config=styling_config,
    cost_tracking=True,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=Topic, default=Topic.WIKIPEDIA_ARTICLE)
    parser.add_argument("--file_dir", type=Path, default=Path("data/wikipedia/files"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/goldens"))
    args = parser.parse_args()

    file_dir = Path(args.file_dir)
    file_paths = list(file_dir.glob("**/*.*"))
    logger.info(f"Found {len(file_paths)} files in {file_dir}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving goldens to {output_dir}")

    for file_path in file_paths:
        logger.info(f"Synthesizing {file_path}")
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=[str(file_path)],
            include_expected_output=True,
            context_construction_config=context_construction_config,
            max_goldens_per_context=1,
        )
        logger.info(f"Synthesis cost: {synthesizer.synthesis_cost}")
        save_goldens_to_files(goldens, output_dir)

    # # Save as JSON with a custom filename my_dataset.json
    # synthesizer.save_as(
    #     file_type="json",
    #     directory="RAG/data/goldens",
    #     file_name=Path(file_path).stem,
    # )
