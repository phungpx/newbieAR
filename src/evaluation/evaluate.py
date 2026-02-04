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
from deepeval.test_case import LLMTestCase

from src.settings import settings
from src.retrieval.rag import Retrieval
from src.evaluation.bedrock_llm_wrapper import BedrockLLMWrapper
from src.evaluation.base_metric_wrapper import BaseMetricWrapper

deepeval.login(settings.confident_api_key)


def create_metrics(
    threshold: float = 0.5,
    include_reason: bool = True,
    async_mode: bool = True,
):
    critique_model = BedrockLLMWrapper(
        model=settings.critique_model_name,
        region_name=settings.critique_model_region_name,
    )

    logger.info(f"Critique model name: {settings.critique_model_name}")
    logger.info(f"Critique model region name: {settings.critique_model_region_name}")

    _metrics = [
        AnswerRelevancyMetric(
            model=critique_model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
        FaithfulnessMetric(
            model=critique_model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
        ContextualPrecisionMetric(
            model=critique_model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
        ContextualRecallMetric(
            model=critique_model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
        ContextualRelevancyMetric(
            model=critique_model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
    ]

    _metric_wrappers = [BaseMetricWrapper(metric) for metric in _metrics]

    return _metric_wrappers


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
        if hasattr(metric.base_metric, "verdicts"):
            verdicts = metric.base_metric.verdicts
        elif hasattr(metric.base_metric, "verdicts_list"):
            verdicts = metric.base_metric.verdicts_list
        else:
            verdicts = None
        metrics_result[metric.__name__] = {
            "score": metric.score,
            "reason": metric.reason,
            "threshold": metric.threshold,
            "is_successful": metric.is_successful(),
            "verdicts": (
                json.loads(
                    json.dumps([verdict.model_dump() for verdict in verdicts], indent=4)
                )
                if verdicts is not None
                else None
            ),
            "token_usage": {
                "prompt_tokens": token_usage.input_tokens,
                "completion_tokens": token_usage.output_tokens,
                "cache_creation_input_tokens": token_usage.cache_creation_input_tokens,
                "cache_read_input_tokens": token_usage.cache_read_input_tokens,
                "turn_name": token_usage.turn_name,
                "model_name": token_usage.model_name,
            },
        }

    # logger.info(f"Metrics result: {json.dumps(metrics_result, indent=4)}")

    return metrics_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, required=True)
    parser.add_argument("--retrieval_window_size", type=int, default=5)
    parser.add_argument(
        "--collection_name", type=str, default=settings.qdrant_collection_name
    )
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include_reason", action="store_true", default=True)
    parser.add_argument("--async_mode", action="store_true", default=True)
    args = parser.parse_args()

    _metric_wrappers = create_metrics(
        threshold=args.threshold,
        include_reason=args.include_reason,
        async_mode=args.async_mode,
    )

    logger.info(
        f"Metrics: {[metric.base_metric.__name__ for metric in _metric_wrappers]}"
    )
    logger.info(f"[..] Metrics threshold: {args.threshold}")
    logger.info(f"[..] Metrics include reason: {args.include_reason}")
    logger.info(f"[..] Metrics async mode: {args.async_mode}")

    for file_path in Path(args.file_dir).glob("*.json"):
        with open(file=file_path, mode="r", encoding="utf-8") as f:
            sample = json.load(f)

        if (
            sample.get("actual_output") is not None
            and sample.get("retrieval_contexts") is not None
            and sample.get("metrics") is not None
            and not args.force_rerun
        ):
            continue

        try:
            test_case, sample = create_llm_test_case(
                file_path=file_path,
                retrieval_window_size=args.retrieval_window_size,
                collection_name=args.collection_name,
            )
            metrics_result = evaluate_llm_test_case_on_metrics(
                test_case=test_case, metrics=_metric_wrappers
            )
            sample["metrics"] = metrics_result
            with open(file=file_path, mode="w", encoding="utf-8") as f:
                json.dump(sample, f, indent=4)

            logger.info(f"Results for {file_path}")
        except Exception as e:
            logger.error(f"Error evaluating {file_path}: {e}")
            continue
