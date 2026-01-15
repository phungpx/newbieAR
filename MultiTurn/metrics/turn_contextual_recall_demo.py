"""
Turn Contextual Recall Metric Demonstration

This script demonstrates the Turn Contextual Recall metric for RAG chatbots.
The metric evaluates whether the retrieval context contains sufficient information
to support the expected outcome throughout a conversation.

Key Concepts:
- Uses ConversationalTestCase with retrieval_context in Turn objects
- Requires expected_outcome for evaluation
- Measures coverage/completeness of retrieved information

Reference: https://deepeval.com/docs/metrics-turn-contextual-recall
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric
from deepeval.models.llms import LocalModel
from settings import ProjectSettings

# Initialize settings and model
settings = ProjectSettings()

model = LocalModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=settings.llm_temperature,
)


# =============================================================================
# Conversation Scenarios
# =============================================================================


def create_complete_recall() -> ConversationalTestCase:
    """
    Conversation where retrieval context covers all expected information.
    Expected: HIGH contextual recall score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Tell me about your return policy."),
            Turn(
                role="assistant",
                content="We offer a 30-day return policy. Items must be unused and in original packaging. Refunds are processed within 5-7 business days to your original payment method.",
                retrieval_context=[
                    "Return Policy: Customers may return items within 30 days of purchase.",
                    "Return Conditions: Items must be unused and in original packaging with all tags attached.",
                    "Refund Processing: Refunds are processed within 5-7 business days after we receive the return.",
                    "Refund Methods: All refunds are issued to the original payment method used for purchase.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain the return timeframe (30 days), conditions (unused, original packaging), refund processing time (5-7 days), and refund method (original payment).",
    )


def create_incomplete_recall() -> ConversationalTestCase:
    """
    Conversation where retrieval context is missing key information.
    Expected: LOW contextual recall score (~0.3-0.4)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Tell me about your return policy."),
            Turn(
                role="assistant",
                content="We accept returns within 30 days.",
                retrieval_context=[
                    # Missing: conditions, refund timing, refund method
                    "Return Policy: Customers may return items within 30 days of purchase.",
                    "Store Hours: Open Monday-Friday 9AM-6PM.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain the return timeframe (30 days), conditions (unused, original packaging), refund processing time (5-7 days), and refund method (original payment).",
    )


def create_partial_recall() -> ConversationalTestCase:
    """
    Conversation where some expected information is covered, some missing.
    Expected: MEDIUM contextual recall score (~0.5-0.7)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are your shipping options?"),
            Turn(
                role="assistant",
                content="We offer standard shipping (5-7 days) for free and express shipping (2-3 days) for $9.99.",
                retrieval_context=[
                    "Shipping Options: Standard shipping is free and takes 5-7 business days.",
                    "Express Shipping: Express shipping costs $9.99 and delivers in 2-3 business days.",
                    # Missing: overnight shipping, international shipping
                ],
            ),
        ],
        expected_outcome="The assistant should explain all shipping options including standard (free, 5-7 days), express ($9.99, 2-3 days), overnight ($19.99, next day), and international options.",
    )


def create_multi_turn_recall() -> ConversationalTestCase:
    """
    Multi-turn conversation testing recall across multiple exchanges.
    Expected: Varies based on context coverage per turn
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What warranty comes with your laptops?"),
            Turn(
                role="assistant",
                content="All laptops come with a 2-year manufacturer warranty covering defects.",
                retrieval_context=[
                    "Laptop Warranty: All laptops include a 2-year manufacturer warranty.",
                    "Warranty Coverage: Covers defects in materials and workmanship under normal use.",
                    "Warranty Exclusions: Does not cover accidental damage or unauthorized modifications.",
                ],
            ),
            Turn(role="user", content="Can I extend the warranty?"),
            Turn(
                role="assistant",
                content="Yes, extended warranty is available.",
                retrieval_context=[
                    # Missing details about extended warranty terms and pricing
                    "Extended Warranty: Extended warranty options are available for purchase.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain the base warranty (2 years, covers defects), exclusions, and extended warranty options including terms, coverage, and pricing.",
    )


def create_comprehensive_coverage() -> ConversationalTestCase:
    """
    Scenario with very comprehensive context covering all aspects.
    Expected: HIGH contextual recall score (~0.95-1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="How do I apply for a store credit card?"),
            Turn(
                role="assistant",
                content="You can apply online or in-store. You need to be 18+ with valid ID. Approval takes about 5 minutes and you get 15% off your first purchase.",
                retrieval_context=[
                    "Credit Card Application: Apply for our store credit card online through our website or in-store at any register.",
                    "Application Requirements: Applicants must be 18 years or older and provide valid government-issued ID.",
                    "Approval Process: Most applications are processed within 5 minutes. You'll receive an instant decision.",
                    "New Cardholder Benefits: Approved applicants receive 15% off their first purchase with the new card.",
                    "Credit Limit: Initial credit limits range from $500 to $5,000 based on creditworthiness.",
                    "APR Information: Standard APR is 24.99%. Special financing available on purchases over $299.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain how to apply (online or in-store), requirements (18+, valid ID), approval time (5 minutes), and new cardholder benefits (15% off first purchase).",
    )


def create_knowledge_gap_scenario() -> ConversationalTestCase:
    """
    Scenario where context has significant gaps in expected coverage.
    Expected: LOW contextual recall score (~0.2-0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What safety features does this car model have?"),
            Turn(
                role="assistant",
                content="The car has multiple airbags and anti-lock brakes.",
                retrieval_context=[
                    "Airbags: The vehicle is equipped with front, side, and curtain airbags.",
                    "ABS: Anti-lock braking system comes standard on all models.",
                    # Missing: many other safety features
                ],
            ),
        ],
        expected_outcome="The assistant should explain all safety features including airbags, ABS, electronic stability control, lane departure warning, blind spot monitoring, automatic emergency braking, and backup camera.",
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_scenario(scenario_name: str, test_case: ConversationalTestCase) -> float:
    """Evaluate a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # Print expected outcome
    print(f"\nExpected Outcome: {test_case.expected_outcome[:120]}...")

    # Print conversation with context
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        print(f"  {i+1}. [{role_label}]: {turn.content[:70]}...")
        if turn.retrieval_context:
            print(f"       Context Documents: {len(turn.retrieval_context)} retrieved")
            for j, ctx in enumerate(turn.retrieval_context[:2]):
                print(f"         - {ctx[:60]}...")
            if len(turn.retrieval_context) > 2:
                print(f"         ... and {len(turn.retrieval_context) - 2} more")

    # Create and run metric
    metric = TurnContextualRecallMetric(
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
    print("TURN CONTEXTUAL RECALL METRIC DEMONSTRATION")
    print("Evaluating Retrieval Coverage in RAG Conversations")
    print("=" * 70)

    results = {}

    # Complete recall
    results["Complete Recall"] = evaluate_scenario(
        "COMPLETE - Context covers all expected info",
        create_complete_recall(),
    )

    # Incomplete recall
    results["Incomplete Recall"] = evaluate_scenario(
        "INCOMPLETE - Context missing key information",
        create_incomplete_recall(),
    )

    # Partial recall
    results["Partial Recall"] = evaluate_scenario(
        "PARTIAL - Some info present, some missing",
        create_partial_recall(),
    )

    # Multi-turn recall
    results["Multi-Turn"] = evaluate_scenario(
        "MULTI-TURN - Recall varies across turns",
        create_multi_turn_recall(),
    )

    # Comprehensive coverage
    results["Comprehensive"] = evaluate_scenario(
        "COMPREHENSIVE - Very thorough context",
        create_comprehensive_coverage(),
    )

    # Knowledge gap
    results["Knowledge Gap"] = evaluate_scenario(
        "KNOWLEDGE GAP - Significant missing info",
        create_knowledge_gap_scenario(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Turn Contextual Recall Metric - Retrieval Coverage Demo")
    print("This demonstrates how DeepEval evaluates retrieval completeness.\n")

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
    print("RECALL FORMULA")
    print("=" * 70)
    print(
        """
    Contextual Recall = Attributable Statements / Total Statements
    
    Process:
    1. Break expected_outcome into individual statements
    2. Check if each statement can be attributed to retrieval context
    3. Calculate ratio of attributable statements
    
    Example:
    Expected: "Return policy is 30 days, items must be unused, refunds take 5-7 days"
    
    Statements:
    1. "Return policy is 30 days"    -> Found in context? YES
    2. "Items must be unused"        -> Found in context? YES  
    3. "Refunds take 5-7 days"       -> Found in context? NO
    
    Recall = 2/3 = 0.67
    """
    )

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Contextual Recall measures if context COVERS all expected information
    2. Unlike Precision (ranking), Recall focuses on COMPLETENESS
    3. Low recall indicates gaps in your knowledge base or retrieval
    4. Requires expected_outcome to define what info should be covered
    5. Critical for ensuring RAG systems have comprehensive information
    
    Best Practices for High Recall:
    - Expand retrieval to include more documents
    - Improve chunking to avoid splitting related information
    - Use query expansion to retrieve for related concepts
    - Audit knowledge base for gaps in coverage
    - Consider using ensemble retrievers
    """
    )


if __name__ == "__main__":
    main()
