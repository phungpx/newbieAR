import json
import deepeval
from loguru import logger
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

# from deepeval.evaluate import evaluate
# from deepeval.evaluate.configs import AsyncConfig
# from deepeval.evaluate.types import EvaluationResult
from deepeval.test_case import LLMTestCase

from src.settings import settings
from src.retrieval.rag import Retrieval
from src.evals.bedrock_llm_wrapper import BedrockModelWrapper
from src.evals.base_metric_wrapper import BaseMetricWrapper

deepeval.login(settings.confident_api_key)

critique_model = BedrockModelWrapper(
    model=settings.critique_model_name,
    region_name=settings.critique_model_region_name,
)

logger.info(f"Critique model name: {settings.critique_model_name}")
logger.info(f"Critique model region name: {settings.critique_model_region_name}")

rag_metrics = [
    AnswerRelevancyMetric(
        model=critique_model, threshold=0.5, include_reason=True, async_mode=True
    ),
    FaithfulnessMetric(
        model=critique_model, threshold=0.5, include_reason=True, async_mode=True
    ),
    ContextualPrecisionMetric(
        model=critique_model, threshold=0.5, include_reason=True, async_mode=True
    ),
    ContextualRecallMetric(
        model=critique_model, threshold=0.5, include_reason=True, async_mode=True
    ),
    ContextualRelevancyMetric(
        model=critique_model, threshold=0.5, include_reason=True, async_mode=True
    ),
]

rag_metric_wrappers = [BaseMetricWrapper(metric) for metric in rag_metrics]

logger.info(f"RAG metrics: {[metric.__name__ for metric in rag_metrics]}")


def create_llm_test_case(
    file_path: str,
    retrieval_window_size: int,
    collection_name: str,
) -> tuple[LLMTestCase, dict]:
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        sample = json.load(f)

    # Create test case with input, expected output, and context
    test_case = LLMTestCase(
        input=sample["input"],
        expected_output=sample["expectedOutput"],
        context=sample["context"],
        additional_metadata={
            "additionalMetadata": sample["additionalMetadata"],
            "sourceFile": sample["sourceFile"],
        },
    )

    # Run RAG to get actual output and retrieval contexts
    retrieval_contexts, actual_output = Retrieval().generate(
        query=test_case.input,
        limit=retrieval_window_size,
        collection_name=collection_name,
    )
    test_case.actual_output = actual_output
    test_case.retrieval_context = [
        retrieval_context["content"] for retrieval_context in retrieval_contexts
    ]

    # Add metrics to sample
    sample["actual_output"] = actual_output
    sample["retrieval_contexts"] = retrieval_contexts

    return test_case, sample


def evaluate_llm_test_case_on_metrics(
    test_case: LLMTestCase,
    metrics: list[BaseMetricWrapper],
) -> dict:
    metrics_result = {}
    for metric in metrics:
        logger.info(f"Evaluating metric: {metric.__name__}")
        metric.measure(test_case, _log_metric_to_confident=True)
        token_usage = metric.get_last_token_usage()
        metrics_result[metric.__name__] = {
            "score": metric.score,
            "reason": metric.reason,
            "threshold": metric.threshold,
            "is_successful": metric.is_successful(),
            "token_usage": {
                "prompt_tokens": token_usage.input_tokens,
                "completion_tokens": token_usage.output_tokens,
                "cache_creation_input_tokens": token_usage.cache_creation_input_tokens,
                "cache_read_input_tokens": token_usage.cache_read_input_tokens,
                "turn_name": token_usage.turn_name,
                "model_name": token_usage.model_name,
            },
        }

    logger.info(f"Metrics result: {json.dumps(metrics_result, indent=4)}")

    return metrics_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, required=True)
    parser.add_argument("--retrieval_window_size", type=int, default=5)
    parser.add_argument(
        "--collection_name", type=str, default=settings.qdrant_collection_name
    )
    args = parser.parse_args()

    for file_path in Path(args.file_dir).glob("*.json"):
        test_case, sample = create_llm_test_case(
            file_path=file_path,
            retrieval_window_size=args.retrieval_window_size,
            collection_name=args.collection_name,
        )
        metrics_result = evaluate_llm_test_case_on_metrics(
            test_case=test_case,
            metrics=rag_metric_wrappers,
        )
        sample["metrics"] = metrics_result
        with open(file=file_path, mode="w", encoding="utf-8") as f:
            json.dump(sample, f, indent=4)

        logger.info(f"Results for {file_path}: {metrics_result}")
