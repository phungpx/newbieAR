"""
Faithfulness Metric Demonstration

This script demonstrates the Faithfulness metric for RAG systems.
The metric evaluates whether the LLM's actual_output factually aligns
with the provided retrieval_context, detecting hallucinations.

Key Concepts:
- Uses LLMTestCase with input, actual_output, and retrieval_context
- Measures if claims in response are supported by context
- Detects hallucinations - information not grounded in context
- A claim is truthful if it does NOT contradict the context

Reference: https://deepeval.com/docs/metrics-faithfulness
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
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


def create_fully_faithful_case() -> LLMTestCase:
    """
    Response is completely grounded in the retrieval context.
    Expected: HIGH faithfulness score (~1.0)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="We offer a 30-day return policy. Items must be unused and in original packaging. You'll receive a full refund once we process the return.",
        retrieval_context=[
            "Return Policy: Customers may return items within 30 days of purchase. Items must be unused and in original packaging. Full refunds are processed within 5-7 business days after the returned item is received.",
        ],
    )


def create_hallucinating_case() -> LLMTestCase:
    """
    Response contains multiple hallucinated claims not in context.
    Expected: LOW faithfulness score (~0.2-0.3)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="We offer a 90-day return policy with no questions asked! You can return items even if they've been used, and we'll give you a full refund plus a 10% bonus credit for your next purchase.",
        retrieval_context=[
            "Return Policy: Customers may return items within 30 days of purchase. Items must be unused and in original packaging. Full refunds are processed within 5-7 business days.",
        ],
    )  # Hallucinations: 90 days (vs 30), no questions asked, used items, 10% bonus


def create_partially_faithful_case() -> LLMTestCase:
    """
    Response has mix of faithful and unfaithful claims.
    Expected: MEDIUM faithfulness score (~0.5-0.6)
    """
    return LLMTestCase(
        input="Tell me about the product warranty.",
        actual_output="The product comes with a 2-year warranty covering manufacturing defects. We also offer free lifetime repairs and a satisfaction guarantee where you can return for any reason within a year.",
        retrieval_context=[
            "Warranty: 2-year limited warranty covers manufacturing defects. Does not cover accidental damage or normal wear and tear.",
        ],
    )  # First claim faithful; "free lifetime repairs" and "year return" hallucinated


def create_contradicting_case() -> LLMTestCase:
    """
    Response directly contradicts information in the context.
    Expected: LOW faithfulness score (~0.2)
    """
    return LLMTestCase(
        input="What are the shipping costs?",
        actual_output="All shipping is completely free with no minimum order required. We ship to all countries worldwide.",
        retrieval_context=[
            "Shipping: Free shipping on orders over $50. Standard shipping costs $5.99 for orders under $50. We currently ship to the US and Canada only.",
        ],
    )  # Contradictions: free shipping (vs $50 minimum), all countries (vs US/Canada only)


def create_extrapolating_case() -> LLMTestCase:
    """
    Response makes reasonable inferences but goes beyond what's stated.
    Expected: MEDIUM-HIGH faithfulness score (~0.6-0.8)
    """
    return LLMTestCase(
        input="How do I contact customer support?",
        actual_output="You can reach our customer support team by phone at 1-800-555-0123 or by email at support@company.com. Our team is available 24/7 and typically responds within an hour.",
        retrieval_context=[
            "Contact Information: Customer service phone: 1-800-555-0123. Email: support@company.com. Hours: Monday-Friday 9AM-6PM EST.",
        ],
    )  # Phone/email correct; "24/7" and "within an hour" are extrapolations


def create_ambiguous_claims_case() -> LLMTestCase:
    """
    Response contains claims that are hard to verify against context.
    Expected: Varies based on penalize_ambiguous_claims setting
    """
    return LLMTestCase(
        input="Is this a good product?",
        actual_output="This is an excellent product with great build quality. Customers love it and it's one of our best sellers. The value for money is outstanding.",
        retrieval_context=[
            "Product Details: Model XYZ-100. Weight: 2.5 lbs. Dimensions: 10x8x3 inches. Material: Aluminum alloy. Color options: Silver, Black, Blue.",
        ],
    )  # Claims about quality/popularity can't be verified from technical specs


def create_multiple_context_case() -> LLMTestCase:
    """
    Response synthesizes information from multiple context documents.
    Expected: HIGH faithfulness score (~0.9-1.0)
    """
    return LLMTestCase(
        input="Tell me about your premium membership.",
        actual_output="Our premium membership costs $9.99 per month and includes free shipping on all orders, early access to sales, and a 15% discount on all purchases.",
        retrieval_context=[
            "Premium Membership Pricing: $9.99/month or $99/year. Cancel anytime.",
            "Premium Benefits: Free shipping on all orders, no minimum required.",
            "Premium Exclusive: Early access to all sales events, 24 hours before general public.",
            "Premium Discount: 15% off all regular-priced items.",
        ],
    )


def create_no_claims_case() -> LLMTestCase:
    """
    Response acknowledges lack of information appropriately.
    Expected: HIGH faithfulness score (~1.0) - no unfaithful claims
    """
    return LLMTestCase(
        input="What are your store hours?",
        actual_output="I don't have specific information about store hours in my current context. I'd recommend checking the store locator on our website or calling customer service at the number provided.",
        retrieval_context=[
            "Contact Information: Customer service phone: 1-800-555-0123. Website: www.company.com",
        ],
    )  # Faithfully acknowledges missing information


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_scenario(
    scenario_name: str,
    test_case: LLMTestCase,
    penalize_ambiguous: bool = False,
) -> float:
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
    context_preview = test_case.retrieval_context[0][:60] + "..."
    print(f"Context: {context_preview}")

    # Create and run metric
    metric = FaithfulnessMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
        penalize_ambiguous_claims=penalize_ambiguous,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Faithfulness Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("FAITHFULNESS METRIC DEMONSTRATION")
    print("Evaluating Response Grounding in Retrieval Context")
    print("=" * 70)

    results = {}

    # Fully faithful
    results["Fully Faithful"] = evaluate_scenario(
        "FULLY FAITHFUL - All claims grounded in context",
        create_fully_faithful_case(),
    )

    # Hallucinating
    results["Hallucinating"] = evaluate_scenario(
        "HALLUCINATING - Multiple fabricated claims",
        create_hallucinating_case(),
    )

    # Partially faithful
    results["Partially Faithful"] = evaluate_scenario(
        "PARTIALLY FAITHFUL - Mix of true and false claims",
        create_partially_faithful_case(),
    )

    # Contradicting
    results["Contradicting"] = evaluate_scenario(
        "CONTRADICTING - Claims contradict context",
        create_contradicting_case(),
    )

    # Extrapolating
    results["Extrapolating"] = evaluate_scenario(
        "EXTRAPOLATING - Goes beyond stated facts",
        create_extrapolating_case(),
    )

    # Ambiguous
    results["Ambiguous Claims"] = evaluate_scenario(
        "AMBIGUOUS - Claims hard to verify",
        create_ambiguous_claims_case(),
    )

    # Multiple context
    results["Multi-Context"] = evaluate_scenario(
        "MULTI-CONTEXT - Synthesizes multiple sources",
        create_multiple_context_case(),
    )

    # No claims
    results["No Claims"] = evaluate_scenario(
        "NO CLAIMS - Acknowledges missing info",
        create_no_claims_case(),
    )

    return results


def run_batch_evaluation():
    """Run batch evaluation with all test cases."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION")
    print("=" * 70)

    test_cases = [
        create_fully_faithful_case(),
        create_hallucinating_case(),
        create_partially_faithful_case(),
        create_contradicting_case(),
        create_extrapolating_case(),
        create_ambiguous_claims_case(),
        create_multiple_context_case(),
        create_no_claims_case(),
    ]

    metric = FaithfulnessMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
    )

    results = evaluate(test_cases=test_cases, metrics=[metric])
    return results


def demonstrate_ambiguous_penalty():
    """Demonstrate the effect of penalize_ambiguous_claims option."""
    print("\n" + "=" * 70)
    print("AMBIGUOUS CLAIMS PENALTY COMPARISON")
    print("=" * 70)

    test_case = create_ambiguous_claims_case()

    print("\n--- Without Penalty ---")
    score_without = evaluate_scenario(
        "Ambiguous claims NOT penalized",
        test_case,
        penalize_ambiguous=False,
    )

    print("\n--- With Penalty ---")
    score_with = evaluate_scenario(
        "Ambiguous claims ARE penalized",
        test_case,
        penalize_ambiguous=True,
    )

    print(f"\nComparison: Without penalty={score_without:.2f}, With penalty={score_with:.2f}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Faithfulness Metric - Hallucination Detection Demo")
    print("This demonstrates how DeepEval evaluates response faithfulness to context.\n")

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
    1. Faithfulness measures if response claims are grounded in retrieval context
    2. A claim is truthful if it does NOT contradict the context
    3. Score = Truthful Claims / Total Claims
    4. Detects hallucinations - information made up by the model
    5. Different from Answer Relevancy (which doesn't require context)
    
    Types of Unfaithful Responses:
    - Fabrication: Making up facts not in context
    - Contradiction: Stating opposite of what context says
    - Exaggeration: Overstating what the context says
    - Extrapolation: Drawing unsupported conclusions
    
    Best Practices for High Faithfulness:
    - Instruct model to only use provided context
    - Train model to say "I don't know" when context is insufficient
    - Use citation mechanisms to ground responses
    - Enable penalize_ambiguous_claims for strict accuracy
    """
    )


if __name__ == "__main__":
    main()
