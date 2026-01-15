"""
Contextual Relevancy Metric Demonstration

This script demonstrates the Contextual Relevancy metric for RAG systems.
The metric evaluates the overall relevance of retrieval_context for a given
input, measuring if retrieved documents are actually useful.

Key Concepts:
- Uses LLMTestCase with input, actual_output, and retrieval_context
- Measures if retrieved documents are relevant to the user's question
- Does NOT require expected_output (referenceless for ground truth)
- Different from Contextual Precision (which measures ranking)

Reference: https://deepeval.com/docs/metrics-contextual-relevancy
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
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


def create_highly_relevant_context_case() -> LLMTestCase:
    """
    All retrieved documents are relevant to the input question.
    Expected: HIGH relevancy score (~1.0)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="You can return items within 30 days for a full refund.",
        retrieval_context=[
            "Return Policy: Items can be returned within 30 days of purchase.",
            "Full refunds are processed for all eligible returns.",
            "Items must be unused and in original packaging.",
            "Original receipt is required for all returns.",
        ],
    )


def create_irrelevant_context_case() -> LLMTestCase:
    """
    None of the retrieved documents are relevant to the question.
    Expected: LOW relevancy score (~0.0-0.2)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="I couldn't find information about the return policy.",
        retrieval_context=[
            "Our company was founded in 1985 by John Smith.",
            "We now have over 500 employees worldwide.",
            "Our headquarters is located in San Francisco.",
            "The company went public in 2010.",
        ],
    )


def create_mixed_relevancy_case() -> LLMTestCase:
    """
    Some documents are relevant, others are not.
    Expected: MEDIUM relevancy score (~0.5)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="Returns are accepted within 30 days.",
        retrieval_context=[
            "Return Policy: Items can be returned within 30 days.",  # Relevant
            "Our stores are open Monday through Friday 9AM-6PM.",  # Irrelevant
            "Refunds are processed within 5-7 business days.",  # Relevant
            "Join our loyalty program for exclusive rewards.",  # Irrelevant
        ],
    )


def create_tangentially_relevant_case() -> LLMTestCase:
    """
    Documents are related to the topic but don't directly answer.
    Expected: MEDIUM relevancy score (~0.4-0.6)
    """
    return LLMTestCase(
        input="Can I get a refund if my product is defective?",
        actual_output="Defective products may be eligible for refund.",
        retrieval_context=[
            "Our products undergo rigorous quality testing.",
            "Customer satisfaction is our top priority.",
            "Contact customer service for product issues.",
            "We stand behind the quality of our products.",
        ],
    )  # Related to quality/service but doesn't answer about refunds


def create_noisy_context_case() -> LLMTestCase:
    """
    One relevant document buried among many irrelevant ones.
    Expected: LOW relevancy score (~0.1-0.2)
    """
    return LLMTestCase(
        input="What is the warranty period?",
        actual_output="Products come with a 2-year warranty.",
        retrieval_context=[
            "Free shipping on orders over $50.",
            "We accept Visa, Mastercard, and PayPal.",
            "Subscribe to our newsletter for updates.",
            "Follow us on social media for promotions.",
            "Warranty: All products include 2-year coverage.",  # Only relevant one
            "Store locations available on our website.",
            "Gift cards available in various denominations.",
        ],
    )


def create_comprehensive_relevant_case() -> LLMTestCase:
    """
    Extensive context that is all highly relevant.
    Expected: HIGH relevancy score (~0.9-1.0)
    """
    return LLMTestCase(
        input="Tell me about your shipping options and costs.",
        actual_output="We offer standard and express shipping with various costs.",
        retrieval_context=[
            "Shipping Options: We offer Standard, Express, and Overnight shipping.",
            "Standard Shipping: Free for orders over $50, otherwise $5.99. Delivery in 5-7 business days.",
            "Express Shipping: $12.99 flat rate. Delivery in 2-3 business days.",
            "Overnight Shipping: $24.99. Order by 2 PM for next-day delivery.",
            "Shipping Coverage: We ship to all 50 US states.",
            "International: Available to Canada and Mexico with additional fees.",
        ],
    )


def create_duplicate_context_case() -> LLMTestCase:
    """
    Context contains duplicate/redundant relevant information.
    Expected: HIGH relevancy score (~1.0) - duplicates don't hurt
    """
    return LLMTestCase(
        input="What are your store hours?",
        actual_output="We're open 9AM to 6PM on weekdays.",
        retrieval_context=[
            "Store Hours: Monday-Friday 9:00 AM - 6:00 PM.",
            "Hours of Operation: Open Monday through Friday, 9 AM to 6 PM.",
            "When are we open? 9AM-6PM, Monday to Friday.",
            "Visit us during our business hours: M-F, 9-6.",
        ],
    )


def create_partial_answer_context_case() -> LLMTestCase:
    """
    Context answers part of a multi-part question.
    Expected: MEDIUM-HIGH relevancy (~0.6-0.8)
    """
    return LLMTestCase(
        input="What is the price and availability of the XYZ product?",
        actual_output="I found price info but not availability.",
        retrieval_context=[
            "XYZ Product Pricing: Base model $299, Pro model $499.",
            "XYZ Features: 10-inch display, 256GB storage, all-day battery.",
            "XYZ Colors: Available in Silver, Space Gray, and Gold.",
            # Missing: Availability/stock information
        ],
    )


def create_technical_irrelevant_case() -> LLMTestCase:
    """
    Context is technical but not relevant to the question.
    Expected: LOW relevancy score (~0.1-0.2)
    """
    return LLMTestCase(
        input="How do I reset my password?",
        actual_output="I couldn't find password reset instructions.",
        retrieval_context=[
            "System Requirements: Windows 10 or macOS 10.14 minimum.",
            "Browser Support: Chrome, Firefox, Safari, Edge.",
            "API Documentation: RESTful API with JSON responses.",
            "SDK: Available for Python, JavaScript, and Java.",
        ],
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
    print(f"\nRetrieval Context ({len(test_case.retrieval_context)} documents):")
    for i, ctx in enumerate(test_case.retrieval_context[:5]):
        ctx_preview = ctx[:55] + "..." if len(ctx) > 55 else ctx
        print(f"  {i+1}. {ctx_preview}")
    if len(test_case.retrieval_context) > 5:
        print(f"  ... and {len(test_case.retrieval_context) - 5} more")

    # Create and run metric
    metric = ContextualRelevancyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Contextual Relevancy Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("CONTEXTUAL RELEVANCY METRIC DEMONSTRATION")
    print("Evaluating Retrieved Context Usefulness")
    print("=" * 70)

    results = {}

    # Highly relevant
    results["Highly Relevant"] = evaluate_scenario(
        "HIGHLY RELEVANT - All context on-topic",
        create_highly_relevant_context_case(),
    )

    # Irrelevant
    results["Irrelevant"] = evaluate_scenario(
        "IRRELEVANT - No useful context retrieved",
        create_irrelevant_context_case(),
    )

    # Mixed relevancy
    results["Mixed Relevancy"] = evaluate_scenario(
        "MIXED - Some relevant, some not",
        create_mixed_relevancy_case(),
    )

    # Tangentially relevant
    results["Tangential"] = evaluate_scenario(
        "TANGENTIAL - Related but doesn't answer",
        create_tangentially_relevant_case(),
    )

    # Noisy context
    results["Noisy Context"] = evaluate_scenario(
        "NOISY - One relevant among many irrelevant",
        create_noisy_context_case(),
    )

    # Comprehensive relevant
    results["Comprehensive"] = evaluate_scenario(
        "COMPREHENSIVE - Extensive relevant context",
        create_comprehensive_relevant_case(),
    )

    # Duplicate context
    results["Duplicate Context"] = evaluate_scenario(
        "DUPLICATE - Redundant relevant info",
        create_duplicate_context_case(),
    )

    # Partial answer
    results["Partial Answer"] = evaluate_scenario(
        "PARTIAL ANSWER - Context only answers part",
        create_partial_answer_context_case(),
    )

    # Technical irrelevant
    results["Tech Irrelevant"] = evaluate_scenario(
        "TECH IRRELEVANT - Wrong technical docs",
        create_technical_irrelevant_case(),
    )

    return results


def run_batch_evaluation():
    """Run batch evaluation with all test cases."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION")
    print("=" * 70)

    test_cases = [
        create_highly_relevant_context_case(),
        create_irrelevant_context_case(),
        create_mixed_relevancy_case(),
        create_tangentially_relevant_case(),
        create_noisy_context_case(),
        create_comprehensive_relevant_case(),
        create_duplicate_context_case(),
        create_partial_answer_context_case(),
        create_technical_irrelevant_case(),
    ]

    metric = ContextualRelevancyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
    )

    results = evaluate(test_cases=test_cases, metrics=[metric])
    return results


def compare_relevancy_vs_precision():
    """Demonstrate difference between Contextual Relevancy and Precision."""
    print("\n" + "=" * 70)
    print("CONTEXTUAL RELEVANCY vs CONTEXTUAL PRECISION")
    print("=" * 70)

    print(
        """
    Key Differences:
    
    CONTEXTUAL RELEVANCY:
    - Measures: Are the retrieved documents relevant to the INPUT?
    - Does NOT require: expected_output
    - Does NOT care about: document ordering
    - Use when: You don't have ground truth expected answers
    
    CONTEXTUAL PRECISION:
    - Measures: Are RELEVANT documents ranked HIGHER than irrelevant?
    - REQUIRES: expected_output (to determine relevance)
    - CARES about: document ordering (list position matters)
    - Use when: You have ground truth and want to evaluate re-ranking
    
    Example:
    - Same context, same input
    - Relevancy: "Is this context useful for this question?"
    - Precision: "Is useful context ranked above useless context?"
    """
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Contextual Relevancy Metric - Context Usefulness Demo")
    print("This demonstrates how DeepEval evaluates retrieved context relevance.\n")

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
    1. Contextual Relevancy measures if retrieved context is useful for the input
    2. Extracts statements from retrieval_context, checks relevance to input
    3. Score = Relevant Statements / Total Statements
    4. Does NOT require expected_output (unlike Precision and Recall)
    5. Does NOT evaluate ranking (unlike Contextual Precision)
    
    When to Use Contextual Relevancy:
    - Quick retriever evaluation without ground truth
    - First-pass check before more detailed metrics
    - When you don't have expected outputs for test cases
    
    Types of Low Relevancy Issues:
    - Wrong topic: Retrieved docs about unrelated subject
    - Tangential: Related but doesn't answer the question
    - Noisy results: Few relevant docs among many irrelevant
    
    Best Practices for High Relevancy:
    - Use semantic search over keyword-only search
    - Fine-tune embeddings for your domain
    - Filter low-relevance results post-retrieval
    - Use hybrid search (semantic + keyword)
    
    Comparison with Other Metrics:
    - Answer Relevancy: Is OUTPUT relevant to INPUT?
    - Contextual Relevancy: Is CONTEXT relevant to INPUT?
    - Contextual Precision: Are relevant docs RANKED higher?
    - Contextual Recall: Did we GET all needed info?
    """
    )


if __name__ == "__main__":
    main()
