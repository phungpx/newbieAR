"""
Contextual Recall Metric Demonstration

This script demonstrates the Contextual Recall metric for RAG systems.
The metric evaluates the extent to which the retrieval_context aligns
with the expected_output (retrieval completeness).

Key Concepts:
- Uses LLMTestCase with input, actual_output, expected_output, retrieval_context
- Measures if all information needed for ideal answer was retrieved
- Extracts statements from expected_output, checks if found in context
- High recall = retrieved all necessary information

Reference: https://deepeval.com/docs/metrics-contextual-recall
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
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


def create_complete_recall_case() -> LLMTestCase:
    """
    All information needed for expected output is in the context.
    Expected: HIGH recall score (~1.0)
    """
    return LLMTestCase(
        input="What is your return and refund policy?",
        actual_output="You can return within 30 days for a full refund.",
        expected_output="Returns are accepted within 30 days. A full refund is processed within 5-7 business days. Original receipt is required.",
        retrieval_context=[
            "Return Policy: Items can be returned within 30 days of purchase.",
            "Refund Processing: All refunds are processed within 5-7 business days after the return is received.",
            "Receipt Requirement: Original receipt or order confirmation is required for all returns.",
        ],
    )


def create_partial_recall_case() -> LLMTestCase:
    """
    Only some information needed for expected output is retrieved.
    Expected: MEDIUM recall score (~0.5-0.6)
    """
    return LLMTestCase(
        input="What is your return and refund policy?",
        actual_output="You can return within 30 days.",
        expected_output="Returns are accepted within 30 days. A full refund is processed within 5-7 business days. Original receipt is required.",
        retrieval_context=[
            "Return Policy: Items can be returned within 30 days of purchase.",
            # Missing: Refund processing time
            # Missing: Receipt requirement
            "Store hours: Monday-Friday 9AM-6PM.",
        ],
    )


def create_no_recall_case() -> LLMTestCase:
    """
    None of the information needed is in the retrieved context.
    Expected: LOW recall score (~0.0)
    """
    return LLMTestCase(
        input="What is your return policy?",
        actual_output="I don't have information about returns.",
        expected_output="30-day return policy with full refund. Receipt required. Items must be unused.",
        retrieval_context=[
            "Our company was founded in 1985.",
            "We have offices in 12 countries.",
            "Our CEO is John Smith.",
            "Stock ticker: COMP",
        ],
    )


def create_multi_fact_complete_case() -> LLMTestCase:
    """
    Expected output has many facts, all found in context.
    Expected: HIGH recall score (~1.0)
    """
    return LLMTestCase(
        input="Tell me everything about the premium membership.",
        actual_output="Premium membership includes many benefits.",
        expected_output="Premium costs $9.99/month. Includes free shipping, 15% discount, early sale access, and priority support.",
        retrieval_context=[
            "Premium Pricing: Monthly subscription costs $9.99, annual costs $99.",
            "Premium Shipping: Free shipping on all orders, no minimum purchase required.",
            "Premium Discount: Members receive 15% off all regular-priced merchandise.",
            "Premium Sales: 24-hour early access to all promotional sales events.",
            "Premium Support: Dedicated priority customer service phone line.",
        ],
    )


def create_multi_fact_partial_case() -> LLMTestCase:
    """
    Expected output has many facts, only some found in context.
    Expected: MEDIUM recall score (~0.4-0.6)
    """
    return LLMTestCase(
        input="Tell me everything about the premium membership.",
        actual_output="Premium membership has some benefits.",
        expected_output="Premium costs $9.99/month. Includes free shipping, 15% discount, early sale access, and priority support.",
        retrieval_context=[
            "Premium Pricing: Monthly subscription costs $9.99, annual costs $99.",
            "Premium Shipping: Free shipping on all orders, no minimum purchase required.",
            # Missing: Discount info
            # Missing: Early sale access
            # Missing: Priority support
            "Our rewards program has over 1 million members.",
        ],
    )


def create_irrelevant_context_case() -> LLMTestCase:
    """
    Context is extensive but not relevant to expected output.
    Expected: LOW recall score (~0.0-0.2)
    """
    return LLMTestCase(
        input="What are the warranty terms?",
        actual_output="The warranty covers defects for 2 years.",
        expected_output="2-year warranty covers manufacturing defects. Does not cover accidental damage. Extended warranty available.",
        retrieval_context=[
            "Product Dimensions: 10 x 8 x 3 inches.",
            "Product Weight: 2.5 lbs.",
            "Available Colors: Silver, Black, Blue.",
            "Material: Aircraft-grade aluminum.",
            "Power: Rechargeable lithium battery.",
        ],
    )


def create_overlapping_context_case() -> LLMTestCase:
    """
    Context has redundant/overlapping information.
    Expected: HIGH recall score (~1.0)
    """
    return LLMTestCase(
        input="How long do returns take?",
        actual_output="Returns are processed within 30 days.",
        expected_output="Return window is 30 days. Refunds processed in 5-7 business days.",
        retrieval_context=[
            "Returns: 30-day return window for all purchases.",
            "Return Period: Customers have 30 days to return items.",
            "Refunds: Processed within 5-7 business days.",
            "Refund Timeline: Allow 5-7 business days for refund processing.",
        ],
    )


def create_implicit_information_case() -> LLMTestCase:
    """
    Context implies but doesn't explicitly state expected information.
    Expected: Varies - depends on how strict the evaluation is
    """
    return LLMTestCase(
        input="Can I return used items?",
        actual_output="Items must be in original condition.",
        expected_output="Items must be unused and in original packaging for returns.",
        retrieval_context=[
            "Return Condition: Items must be in their original, unopened packaging.",
            "Return Quality: Products should be in resalable condition.",
            # "Unused" is implied but not explicit
        ],
    )


def create_complex_expected_output_case() -> LLMTestCase:
    """
    Expected output contains complex, multi-part statements.
    Expected: HIGH recall if context covers all parts (~0.9-1.0)
    """
    return LLMTestCase(
        input="Explain the shipping options and costs.",
        actual_output="Multiple shipping options are available.",
        expected_output="Standard shipping is free for orders over $50 and takes 5-7 days. Express shipping costs $12.99 and delivers in 2-3 days. International shipping is available to 30 countries.",
        retrieval_context=[
            "Standard Shipping: Free for orders over $50. Delivery time: 5-7 business days.",
            "Express Shipping: $12.99 flat rate. Delivery time: 2-3 business days.",
            "International: We ship to 30 countries. Delivery times and costs vary by destination.",
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
    print(f"Expected Output: {test_case.expected_output[:100]}...")
    print(f"\nRetrieval Context ({len(test_case.retrieval_context)} documents):")
    for i, ctx in enumerate(test_case.retrieval_context[:4]):  # Show first 4
        ctx_preview = ctx[:50] + "..." if len(ctx) > 50 else ctx
        print(f"  {i+1}. {ctx_preview}")
    if len(test_case.retrieval_context) > 4:
        print(f"  ... and {len(test_case.retrieval_context) - 4} more")

    # Create and run metric
    metric = ContextualRecallMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Contextual Recall Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("CONTEXTUAL RECALL METRIC DEMONSTRATION")
    print("Evaluating Retrieval Completeness")
    print("=" * 70)

    results = {}

    # Complete recall
    results["Complete Recall"] = evaluate_scenario(
        "COMPLETE RECALL - All needed info retrieved",
        create_complete_recall_case(),
    )

    # Partial recall
    results["Partial Recall"] = evaluate_scenario(
        "PARTIAL RECALL - Some info missing",
        create_partial_recall_case(),
    )

    # No recall
    results["No Recall"] = evaluate_scenario(
        "NO RECALL - None of needed info retrieved",
        create_no_recall_case(),
    )

    # Multi-fact complete
    results["Multi-Fact Complete"] = evaluate_scenario(
        "MULTI-FACT COMPLETE - Many facts, all found",
        create_multi_fact_complete_case(),
    )

    # Multi-fact partial
    results["Multi-Fact Partial"] = evaluate_scenario(
        "MULTI-FACT PARTIAL - Many facts, some missing",
        create_multi_fact_partial_case(),
    )

    # Irrelevant context
    results["Irrelevant Context"] = evaluate_scenario(
        "IRRELEVANT CONTEXT - Extensive but wrong info",
        create_irrelevant_context_case(),
    )

    # Overlapping context
    results["Overlapping Context"] = evaluate_scenario(
        "OVERLAPPING - Redundant info still counts",
        create_overlapping_context_case(),
    )

    # Complex expected
    results["Complex Expected"] = evaluate_scenario(
        "COMPLEX - Multi-part expected output",
        create_complex_expected_output_case(),
    )

    return results


def run_batch_evaluation():
    """Run batch evaluation with all test cases."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION")
    print("=" * 70)

    test_cases = [
        create_complete_recall_case(),
        create_partial_recall_case(),
        create_no_recall_case(),
        create_multi_fact_complete_case(),
        create_multi_fact_partial_case(),
        create_irrelevant_context_case(),
        create_overlapping_context_case(),
        create_complex_expected_output_case(),
    ]

    metric = ContextualRecallMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
    )

    results = evaluate(test_cases=test_cases, metrics=[metric])
    return results


def demonstrate_expected_output_importance():
    """Show how expected_output determines what's measured."""
    print("\n" + "=" * 70)
    print("EXPECTED OUTPUT IMPORTANCE DEMONSTRATION")
    print("Same context, different expected outputs")
    print("=" * 70)

    context = [
        "Return Policy: 30-day returns accepted.",
        "Refunds processed in 5-7 days.",
    ]

    # Simple expected output - easy to satisfy
    simple_case = LLMTestCase(
        input="Return policy?",
        actual_output="30-day returns.",
        expected_output="30-day return policy.",
        retrieval_context=context,
    )

    # Complex expected output - harder to satisfy with same context
    complex_case = LLMTestCase(
        input="Return policy?",
        actual_output="30-day returns.",
        expected_output="30-day returns with full refund. Refunds in 5-7 days. Receipt required. Items must be unused in original packaging.",
        retrieval_context=context,
    )

    print("\n--- SIMPLE EXPECTED OUTPUT ---")
    simple_score = evaluate_scenario(
        "Simple expected: '30-day return policy'",
        simple_case,
    )

    print("\n--- COMPLEX EXPECTED OUTPUT ---")
    complex_score = evaluate_scenario(
        "Complex expected: multiple requirements",
        complex_case,
    )

    print(
        f"\n*** Same context, different scores: Simple={simple_score:.2f}, Complex={complex_score:.2f} ***"
    )
    print("The expected_output determines what information needs to be retrieved!")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Contextual Recall Metric - Retrieval Completeness Demo")
    print("This demonstrates how DeepEval evaluates retrieval coverage.\n")

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
    1. Contextual Recall measures if ALL needed information was retrieved
    2. Extracts statements from expected_output (not actual_output!)
    3. Checks if each statement can be attributed to retrieval_context
    4. Score = Attributable Statements / Total Statements
    5. High recall = retriever captures all relevant information
    
    Why expected_output is used:
    - We're measuring retrieval quality for the IDEAL answer
    - If info for ideal answer isn't retrieved, retriever failed
    - Actual_output may be incomplete due to generation issues
    
    Types of Low Recall Issues:
    - Missing documents: Relevant docs not in knowledge base
    - Poor embedding: Semantically similar docs not retrieved
    - Low top-k: Not enough documents retrieved
    - Query mismatch: Query doesn't match document terms
    
    Best Practices for High Recall:
    - Increase retrieval top-k for better coverage
    - Use hybrid search (semantic + keyword)
    - Optimize chunk sizes to capture full context
    - Use query expansion techniques
    """
    )


if __name__ == "__main__":
    main()
