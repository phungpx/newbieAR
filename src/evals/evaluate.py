import json
import deepeval
from tqdm import tqdm
from loguru import logger
from typing import Any
from pathlib import Path
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from src.evals.bedrock_model import BedrockModelWrapper
from deepeval.evaluate.configs import AsyncConfig

from src.retrieval.rag import Retrieval
from src.settings import settings


def load_json(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def create_deepeval_dataset(
    evaluation_samples: list[dict[str, Any]],
    retrieval_window_size: int,
    collection_name: str,
) -> list[LLMTestCase]:
    logger.info(f"Loaded {len(evaluation_samples)} questions")

    test_cases = []
    for sample in tqdm(evaluation_samples):
        input_question = sample["input"]
        expected_output = sample["expectedOutput"]
        context = sample["context"]
        retrieval_contexts, actual_output = Retrieval().generate(
            query=input_question,
            limit=retrieval_window_size,
            collection_name=collection_name,
        )
        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=[
                retrieval_context["content"] for retrieval_context in retrieval_contexts
            ],
        )
        test_cases.append(test_case)

    logger.info(f"Created {len(test_cases)} test cases")

    return test_cases


retrieval_window_size = 5
file_dirs = [
    "data/goldens/docling",
    "data/goldens/deepseek-ocrv2",
]
evaluation_samples = []
for file_dir in file_dirs:
    for file_path in Path(file_dir).glob("*.json"):
        evaluation_samples.append(load_json(file_path))

test_cases = create_deepeval_dataset(
    evaluation_samples=evaluation_samples,
    retrieval_window_size=retrieval_window_size,
    collection_name=settings.qdrant_collection_name,
)

# evaluator = GPTModel(
#     model=settings.llm_model,
#     api_key=settings.llm_api_key,
#     base_url=settings.llm_base_url,
# )
evaluator = BedrockModelWrapper(
    model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="ap-southeast-1",
)

deepeval.login(settings.confident_api_key)

deepeval.evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        FaithfulnessMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        ContextualPrecisionMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        ContextualRecallMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
        ContextualRelevancyMetric(
            model=evaluator, threshold=0.5, include_reason=True, async_mode=False
        ),
    ],
    async_config=AsyncConfig(run_async=False),
)
