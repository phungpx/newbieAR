"""
Contextual Precision Metric Demonstration

This script demonstrates the Contextual Precision metric for RAG systems.
The metric evaluates whether relevant nodes in retrieval_context are
ranked higher than irrelevant ones (re-ranking quality).

Key Concepts:
- Uses LLMTestCase with input, actual_output, expected_output, retrieval_context
- Measures if relevant context nodes appear before irrelevant ones
- Uses Weighted Cumulative Precision to emphasize top-ranked results
- Order of retrieval_context list IS the ranking being evaluated

Reference: https://deepeval.com/docs/metrics-contextual-precision
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
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


def create_perfect_ranking_case() -> LLMTestCase:
    """
    All relevant documents ranked before irrelevant ones.
    Expected: HIGH precision score (~1.0)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="You can return items within 30 days for a full refund.",
        expected_output="Returns are accepted within 30 days. Full refund with original receipt. Items must be unused.",
        retrieval_context=[
            # Position 1: Relevant
            "Return Policy: Items can be returned within 30 days of purchase.",
            # Position 2: Relevant
            "Full refunds are processed for all eligible returns with original receipt.",
            # Position 3: Relevant
            "Items must be unused and in original packaging for returns.",
            # Position 4: Irrelevant
            "Our company was founded in 1990 in Seattle.",
            # Position 5: Irrelevant
            "We have over 500 retail locations nationwide.",
        ],
    )


def create_worst_ranking_case() -> LLMTestCase:
    """
    All relevant documents ranked at the bottom.
    Expected: LOW precision score (~0.2-0.3)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="You can return items within 30 days for a full refund.",
        expected_output="Returns are accepted within 30 days. Full refund with original receipt. Items must be unused.",
        retrieval_context=[
            # Position 1: Irrelevant
            "Our company was founded in 1990 in Seattle.",
            # Position 2: Irrelevant
            "We have over 500 retail locations nationwide.",
            # Position 3: Irrelevant
            "Join our loyalty program for exclusive rewards.",
            # Position 4: Relevant (bad - should be higher!)
            "Return Policy: Items can be returned within 30 days of purchase.",
            # Position 5: Relevant (bad - should be higher!)
            "Full refunds are processed for all eligible returns with original receipt.",
        ],
    )


def create_mixed_ranking_case() -> LLMTestCase:
    """
    Relevant and irrelevant documents interspersed.
    Expected: MEDIUM precision score (~0.5-0.7)
    """
    return LLMTestCase(
        input="What shipping options do you offer?",
        actual_output="We offer standard and express shipping.",
        expected_output="Standard shipping: 5-7 days, free over $50. Express shipping: 2-3 days, $12.99.",
        retrieval_context=[
            # Position 1: Relevant (good)
            "Standard shipping takes 5-7 business days and is free for orders over $50.",
            # Position 2: Irrelevant
            "Store hours: Monday-Friday 9AM-6PM, Saturday 10AM-4PM.",
            # Position 3: Relevant (ok)
            "Express shipping delivers in 2-3 business days for $12.99.",
            # Position 4: Irrelevant
            "Contact customer service at 1-800-555-0123.",
            # Position 5: Irrelevant
            "We accept Visa, Mastercard, and American Express.",
        ],
    )


def create_single_relevant_top_case() -> LLMTestCase:
    """
    Only one relevant document, but it's at the top.
    Expected: HIGH precision score (~1.0)
    """
    return LLMTestCase(
        input="What is the warranty period?",
        actual_output="The warranty is 2 years.",
        expected_output="2-year warranty covering manufacturing defects.",
        retrieval_context=[
            # Position 1: Relevant (perfect!)
            "Warranty: 2-year coverage for manufacturing defects.",
            # Position 2: Irrelevant
            "Product weight: 2.5 lbs.",
            # Position 3: Irrelevant
            "Available in silver, black, and blue colors.",
            # Position 4: Irrelevant
            "Dimensions: 10 x 8 x 3 inches.",
        ],
    )


def create_single_relevant_bottom_case() -> LLMTestCase:
    """
    Only one relevant document, buried at the bottom.
    Expected: LOW precision score (~0.25)
    """
    return LLMTestCase(
        input="What is the warranty period?",
        actual_output="The warranty is 2 years.",
        expected_output="2-year warranty covering manufacturing defects.",
        retrieval_context=[
            # Position 1: Irrelevant
            "Product weight: 2.5 lbs.",
            # Position 2: Irrelevant
            "Available in silver, black, and blue colors.",
            # Position 3: Irrelevant
            "Dimensions: 10 x 8 x 3 inches.",
            # Position 4: Relevant (bad position!)
            "Warranty: 2-year coverage for manufacturing defects.",
        ],
    )


def create_all_relevant_case() -> LLMTestCase:
    """
    All context documents are relevant.
    Expected: HIGH precision score (~1.0)
    """
    return LLMTestCase(
        input="Tell me about the premium membership benefits.",
        actual_output="Premium includes free shipping, 15% discount, and early sale access.",
        expected_output="Premium membership offers free shipping, 15% off, and 24-hour early access to sales.",
        retrieval_context=[
            "Premium Membership: Free shipping on all orders, no minimum.",
            "Premium Discount: 15% off all regular-priced items.",
            "Premium Exclusive: 24-hour early access to all sales events.",
            "Premium Support: Priority customer service line.",
        ],
    )


def create_all_irrelevant_case() -> LLMTestCase:
    """
    No context documents are relevant to the expected output.
    Expected: Score depends on implementation (typically 0 or undefined)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="I'm not sure about the return policy.",
        expected_output="30-day returns with full refund and receipt required.",
        retrieval_context=[
            "Our CEO received the Innovation Award in 2020.",
            "Company headquarters is in San Francisco.",
            "We employ over 1000 people globally.",
            "Our stock ticker symbol is COMP.",
        ],
    )


def create_many_documents_case() -> LLMTestCase:
    """
    Many documents with varying relevance and positions.
    Expected: MEDIUM precision score (depends on exact ranking)
    """
    return LLMTestCase(
        input="How do I track my order?",
        actual_output="You can track your order using the tracking number in your confirmation email.",
        expected_output="Track orders via email link, website order page, or mobile app with tracking number.",
        retrieval_context=[
            # Position 1: Relevant
            "Order Tracking: Use the tracking number in your confirmation email to track your package.",
            # Position 2: Irrelevant
            "Gift wrapping is available for $4.99 per item.",
            # Position 3: Relevant
            "Track orders on our website by logging into your account and viewing order history.",
            # Position 4: Irrelevant
            "We ship with UPS, FedEx, and USPS depending on your location.",
            # Position 5: Relevant
            "Download our mobile app for real-time order tracking notifications.",
            # Position 6: Irrelevant
            "International shipping may take 2-4 weeks depending on customs.",
            # Position 7: Irrelevant
            "Expedited shipping available for time-sensitive orders.",
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
    print(f"Expected: {test_case.expected_output[:80]}...")
    print(f"\nRetrieval Context (ranked order):")
    for i, ctx in enumerate(test_case.retrieval_context):
        ctx_preview = ctx[:50] + "..." if len(ctx) > 50 else ctx
        print(f"  {i+1}. {ctx_preview}")

    # Create and run metric
    metric = ContextualPrecisionMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Contextual Precision Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("CONTEXTUAL PRECISION METRIC DEMONSTRATION")
    print("Evaluating Re-Ranking Quality of Retrieved Documents")
    print("=" * 70)

    results = {}

    # Perfect ranking
    results["Perfect Ranking"] = evaluate_scenario(
        "PERFECT RANKING - All relevant docs first",
        create_perfect_ranking_case(),
    )

    # Worst ranking
    results["Worst Ranking"] = evaluate_scenario(
        "WORST RANKING - Relevant docs at bottom",
        create_worst_ranking_case(),
    )

    # Mixed ranking
    results["Mixed Ranking"] = evaluate_scenario(
        "MIXED RANKING - Relevant and irrelevant interspersed",
        create_mixed_ranking_case(),
    )

    # Single relevant top
    results["Single Top"] = evaluate_scenario(
        "SINGLE RELEVANT TOP - One relevant doc at position 1",
        create_single_relevant_top_case(),
    )

    # Single relevant bottom
    results["Single Bottom"] = evaluate_scenario(
        "SINGLE RELEVANT BOTTOM - One relevant doc at last position",
        create_single_relevant_bottom_case(),
    )

    # All relevant
    results["All Relevant"] = evaluate_scenario(
        "ALL RELEVANT - Every context doc is relevant",
        create_all_relevant_case(),
    )

    # Many documents
    results["Many Docs"] = evaluate_scenario(
        "MANY DOCUMENTS - Multiple docs with varied ranking",
        create_many_documents_case(),
    )

    return results


def run_batch_evaluation():
    """Run batch evaluation with all test cases."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION")
    print("=" * 70)

    test_cases = [
        create_perfect_ranking_case(),
        create_worst_ranking_case(),
        create_mixed_ranking_case(),
        create_single_relevant_top_case(),
        create_single_relevant_bottom_case(),
        create_all_relevant_case(),
        create_many_documents_case(),
    ]

    metric = ContextualPrecisionMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
    )

    results = evaluate(test_cases=test_cases, metrics=[metric])
    return results


def demonstrate_ranking_impact():
    """Demonstrate how document order affects the score."""
    print("\n" + "=" * 70)
    print("RANKING IMPACT DEMONSTRATION")
    print("Same documents, different order")
    print("=" * 70)

    base_relevant = [
        "Return Policy: 30-day returns accepted.",
        "Full refund with original receipt.",
    ]
    base_irrelevant = [
        "Company founded in 1990.",
        "500 retail locations.",
    ]

    # Best order: relevant first
    best_case = LLMTestCase(
        input="Return policy?",
        actual_output="30-day returns with refund.",
        expected_output="30-day returns, full refund with receipt.",
        retrieval_context=base_relevant + base_irrelevant,
    )

    # Worst order: irrelevant first
    worst_case = LLMTestCase(
        input="Return policy?",
        actual_output="30-day returns with refund.",
        expected_output="30-day returns, full refund with receipt.",
        retrieval_context=base_irrelevant + base_relevant,
    )

    print("\n--- BEST ORDER (relevant first) ---")
    best_score = evaluate_scenario("Relevant docs ranked 1-2", best_case)

    print("\n--- WORST ORDER (irrelevant first) ---")
    worst_score = evaluate_scenario("Relevant docs ranked 3-4", worst_case)

    print(f"\n*** Score Impact: Best={best_score:.2f}, Worst={worst_score:.2f} ***")
    print("Same documents, but order dramatically affects the score!")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Contextual Precision Metric - Re-Ranking Quality Demo")
    print("This demonstrates how DeepEval evaluates document ranking in RAG.\n")

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
    1. Contextual Precision evaluates the RANKING of retrieved documents
    2. Uses Weighted Cumulative Precision - top positions matter more
    3. Requires expected_output to determine what's "relevant"
    4. Order of retrieval_context list IS the ranking being evaluated
    5. LLMs pay more attention to earlier context - ranking matters!
    
    Why Ranking Matters:
    - LLMs have limited attention span for long contexts
    - Top-ranked documents get more attention
    - Poor ranking can cause hallucinations even with good documents
    
    Types of Ranking Issues:
    - Relevant buried: Good docs at bottom of list
    - Noise at top: Irrelevant docs ranked first
    - Interspersed: Relevant docs scattered among irrelevant
    
    Best Practices for High Precision:
    - Use cross-encoder re-rankers after initial retrieval
    - Tune retrieval score thresholds
    - Implement query expansion for better initial retrieval
    - Filter low-relevance documents before generation
    """
    )


if __name__ == "__main__":
    main()
