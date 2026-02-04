from loguru import logger
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from src.models.token_usage import TokenUsage
from src.evaluation.bedrock_llm_wrapper import BedrockLLMWrapper


class BaseMetricWrapper(BaseMetric):
    def __init__(self, base_metric: BaseMetric):
        super().__init__()
        self.base_metric = base_metric
        self.last_token_usage: TokenUsage | None = None
        self.threshold = base_metric.threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """
        Run the base metric and capture token usage.

        Args:
            test_case: The LLM test case to evaluate
            *args: Additional positional arguments for the metric
            **kwargs: Additional keyword arguments for the metric

        Returns:
            The metric score
        """
        # Run the base metric
        result = self.base_metric.measure(test_case, *args, **kwargs)
        self.last_token_usage = self.get_last_token_usage()
        return result

    def get_last_token_usage(self) -> TokenUsage:
        """
        Extract token usage from the base metric's model.

        Supports multiple token tracking patterns:
        - BedrockModelWrapper: model.tracker._current_turn_tokens
        - Direct tracking: model._current_turn_tokens

        Returns:
            TokenUsage with captured tokens or zeros
        """
        try:
            model: BedrockLLMWrapper = self.base_metric.model
            token_usage = model.token_history[-1]
            if token_usage:
                logger.debug(
                    f"Captured tokens from {self.base_metric.__class__.__name__} "
                    f"input={token_usage.input_tokens}, "
                    f"output={token_usage.output_tokens}, "
                    f"model={token_usage.model_name}"
                )
                return token_usage

        except Exception as e:
            logger.warning(f"Failed to extract token usage: {e}")

        # Return zero tokens if extraction fails
        return self._zero_tokens()

    def _zero_tokens(self) -> TokenUsage:
        """Return a TokenUsage with all zeros."""
        return TokenUsage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            model_name=self.base_metric.model.name,
        )

    def get_token_usage(self) -> TokenUsage | None:
        """Get the last captured token usage.
        Returns:
            TokenUsage from last measure() call, or None if not yet measured
        """
        return self.last_token_usage

    def clear_tokens(self) -> None:
        """Reset token tracking."""
        self.last_token_usage = None

    @property
    def score(self) -> float:
        """Delegate score to base metric."""
        return self.base_metric.score

    @property
    def reason(self) -> str:
        """Delegate reason to base metric."""
        return self.base_metric.reason

    def is_successful(self) -> bool:
        """Delegate success check to base metric."""
        return self.base_metric.is_successful()

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Async version of measure.
        Args:
            test_case: The LLM test case to evaluate
            *args: Additional positional arguments for the metric
            **kwargs: Additional keyword arguments for the metric

        Returns:
            The metric score
        """
        # Run the base metric's async measure if available
        if hasattr(self.base_metric, "a_measure"):
            result = await self.base_metric.a_measure(test_case, *args, **kwargs)
        else:
            # Fallback to sync measure
            result = self.base_metric.measure(test_case, *args, **kwargs)

        # Capture token usage
        self.last_token_usage = self.get_last_token_usage()

        return result

    @property
    def __name__(self) -> str:
        return self.base_metric.__name__
