"""
Answer Relevancy Metric Demonstration

This script demonstrates the Answer Relevancy metric for RAG systems.
The metric evaluates how relevant the LLM's actual_output is compared
to the provided input, regardless of retrieval context.

Key Concepts:
- Uses LLMTestCase with input and actual_output
- Measures if response addresses the user's question
- Referenceless - no expected_output or retrieval_context required

Reference: https://deepeval.com/docs/metrics-answer-relevancy
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.llms import LocalModel
from src.settings import ProjectSettings

# Initialize settings and model
settings = ProjectSettings()

model = LocalModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=settings.llm_temperature,
)


# =============================================================================
# Test Case Scenarios
# =============================================================================


def create_highly_relevant_case() -> LLMTestCase:
    """
    Response directly and completely addresses the question.
    Expected: HIGH relevancy score (~1.0)
    """
    return LLMTestCase(
        input="What are your store hours?",
        actual_output="Our store is open Monday through Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM. We are closed on Sundays.",
    )


def create_irrelevant_case() -> LLMTestCase:
    """
    Response doesn't address the question at all.
    Expected: LOW relevancy score (~0.2)
    """
    return LLMTestCase(
        input="What are your store hours?",
        actual_output="Our company was founded in 1985 by John Smith. We have grown to over 500 employees and operate in 12 countries worldwide. Our headquarters is located in San Francisco.",
    )


def create_partially_relevant_case() -> LLMTestCase:
    """
    Response contains relevant information but also off-topic content.
    Expected: MEDIUM relevancy score (~0.5-0.6)
    """
    return LLMTestCase(
        input="What are your store hours?",
        actual_output="We're open 9 AM to 6 PM on weekdays. By the way, we're having a 20% off sale this weekend on all electronics! Our new product line just launched, and our CEO was recently featured in Forbes magazine.",
    )


def create_tangential_case() -> LLMTestCase:
    """
    Response is related to the topic but doesn't directly answer.
    Expected: LOW-MEDIUM relevancy score (~0.3-0.4)
    """
    return LLMTestCase(
        input="What are your store hours?",
        actual_output="You can always check our website for the most up-to-date information. We also have a mobile app where you can see store details. Many customers find it convenient to call ahead before visiting.",
    )


def create_verbose_but_relevant_case() -> LLMTestCase:
    """
    Response is verbose but all content is relevant.
    Expected: HIGH relevancy score (~0.9-1.0)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="""Our return policy is designed to ensure customer satisfaction. Here are the key points:

1. Time Frame: You have 30 days from the date of purchase to return items.
2. Condition: Items must be unused and in original packaging.
3. Receipt: Please bring your original receipt or order confirmation.
4. Refund Method: Refunds are issued to the original payment method.
5. Processing Time: Refunds are typically processed within 5-7 business days.
6. Exceptions: Sale items and personalized products cannot be returned.

If you have any questions about a specific return, please contact our customer service team.""",
    )


def create_question_answer_mismatch_case() -> LLMTestCase:
    """
    Response answers a different question entirely.
    Expected: VERY LOW relevancy score (~0.1)
    """
    return LLMTestCase(
        input="How do I reset my password?",
        actual_output="Our shipping rates are based on package weight and destination. Standard shipping is free for orders over $50. Express shipping costs $12.99 and delivers within 2-3 business days.",
    )


def create_brief_relevant_case() -> LLMTestCase:
    """
    Response is brief but directly answers the question.
    Expected: HIGH relevancy score (~1.0)
    """
    return LLMTestCase(
        input="Do you offer free shipping?",
        actual_output="Yes, we offer free shipping on all orders over $50.",
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_scenario(scenario_name: str, test_case: LLMTestCase) -> float:
    """Evaluate a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # Print test case details
    print(f"\nInput: {test_case.input}")
    output_preview = (
        test_case.actual_output[:100] + "..."
        if len(test_case.actual_output) > 100
        else test_case.actual_output
    )
    print(f"Output: {output_preview}")

    # Create and run metric
    metric = AnswerRelevancyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Answer Relevancy Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("ANSWER RELEVANCY METRIC DEMONSTRATION")
    print("Evaluating Response Relevance to User Input")
    print("=" * 70)

    results = {}

    # Highly relevant
    results["Highly Relevant"] = evaluate_scenario(
        "HIGHLY RELEVANT - Direct answer to question",
        create_highly_relevant_case(),
    )

    # Irrelevant
    results["Irrelevant"] = evaluate_scenario(
        "IRRELEVANT - Response doesn't address question",
        create_irrelevant_case(),
    )

    # Partially relevant
    results["Partially Relevant"] = evaluate_scenario(
        "PARTIALLY RELEVANT - Mix of on-topic and off-topic",
        create_partially_relevant_case(),
    )

    # Tangential
    results["Tangential"] = evaluate_scenario(
        "TANGENTIAL - Related but doesn't answer",
        create_tangential_case(),
    )

    # Verbose but relevant
    results["Verbose Relevant"] = evaluate_scenario(
        "VERBOSE BUT RELEVANT - Long but all on-topic",
        create_verbose_but_relevant_case(),
    )

    # Question mismatch
    results["Question Mismatch"] = evaluate_scenario(
        "QUESTION MISMATCH - Answers different question",
        create_question_answer_mismatch_case(),
    )

    # Brief relevant
    results["Brief Relevant"] = evaluate_scenario(
        "BRIEF RELEVANT - Short but complete answer",
        create_brief_relevant_case(),
    )

    return results


def run_batch_evaluation():
    """Run batch evaluation with all test cases."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION")
    print("=" * 70)

    test_cases = [
        create_highly_relevant_case(),
        create_irrelevant_case(),
        create_partially_relevant_case(),
        create_tangential_case(),
        create_verbose_but_relevant_case(),
        create_question_answer_mismatch_case(),
        create_brief_relevant_case(),
    ]

    metric = AnswerRelevancyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
    )

    results = evaluate(test_cases=test_cases, metrics=[metric])
    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Answer Relevancy Metric - Response Quality Demo")
    print("This demonstrates how DeepEval evaluates response relevance to input.\n")

    # Run all scenarios
    results = run_all_scenarios()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for scenario, score in results.items():
        status = "PASS" if score >= 0.5 else "FAIL"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{scenario:20} [{bar}] {score:.2f} [{status}]")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Answer Relevancy measures if the response addresses the user's question
    2. Extracts statements from actual_output and checks relevance to input
    3. Score = Relevant Statements / Total Statements
    4. Does NOT require retrieval_context or expected_output (referenceless)
    5. Different from Faithfulness (which checks accuracy against context)
    
    Types of Low Relevancy Responses:
    - Off-topic: Response about unrelated subject
    - Tangential: Related but doesn't answer the question
    - Partial: Some relevant, some irrelevant content
    - Mismatch: Answers a different question
    
    Best Practices for High Relevancy:
    - Instruct model to directly answer questions
    - Use structured output formats
    - Avoid promotional/filler content
    - Keep responses focused on the question
    """
    )


if __name__ == "__main__":
    main()
