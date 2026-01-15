from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models.llms import LocalModel
from src.settings import ProjectSettings

settings = ProjectSettings()

model = LocalModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=settings.llm_temperature,
    # max_tokens=settings.llm_max_tokens,
)

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = [
    "All customers are eligible for a 30 day full refund at no extra cost."
]

metric = FaithfulnessMetric(
    model=model,
    threshold=0.7,
    include_reason=True,
)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context,
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

result = evaluate(test_cases=[test_case], metrics=[metric])
print(result)
