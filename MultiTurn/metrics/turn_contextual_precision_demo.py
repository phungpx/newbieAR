"""
Turn Contextual Precision Metric Demonstration

This script demonstrates the Turn Contextual Precision metric for RAG chatbots.
The metric evaluates whether relevant nodes in the retrieval context are ranked
higher than irrelevant nodes throughout a conversation.

Key Concepts:
- Uses ConversationalTestCase with retrieval_context in Turn objects
- Requires expected_outcome for evaluation
- Evaluates ranking quality of retrieved documents

Reference: https://deepeval.com/docs/metrics-turn-contextual-precision
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric
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


def create_well_ranked_context() -> ConversationalTestCase:
    """
    Conversation where relevant documents are ranked first.
    Expected: HIGH contextual precision score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's your return policy for electronics?"),
            Turn(
                role="assistant",
                content="For electronics, we offer a 30-day return policy. Items must be unused and in original packaging with all accessories.",
                retrieval_context=[
                    # Rank 1 - Highly relevant
                    "Electronics Return Policy: All electronics can be returned within 30 days of purchase. Items must be unused, in original packaging, with all accessories and manuals included.",
                    # Rank 2 - Relevant
                    "Return Process: To initiate a return, bring your item and receipt to any store location or start a return online through your account.",
                    # Rank 3 - Less relevant
                    "Store Locations: We have 50 stores nationwide. Visit our website for the nearest location.",
                    # Rank 4 - Irrelevant
                    "Career Opportunities: Join our team! We're hiring sales associates and warehouse staff.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain the return policy for electronics including timeframe, conditions, and requirements.",
    )


def create_poorly_ranked_context() -> ConversationalTestCase:
    """
    Conversation where irrelevant documents are ranked first.
    Expected: LOW contextual precision score (~0.3-0.4)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's your return policy for electronics?"),
            Turn(
                role="assistant",
                content="For electronics, we offer a 30-day return policy with full refund.",
                retrieval_context=[
                    # Rank 1 - Irrelevant
                    "Company History: Founded in 1985, we've grown to become a leading retailer with over 500 employees.",
                    # Rank 2 - Irrelevant
                    "Job Openings: We're currently hiring for customer service and warehouse positions.",
                    # Rank 3 - Irrelevant
                    "Holiday Hours: During the holiday season, stores are open 8AM-10PM.",
                    # Rank 4 - Relevant (but ranked last)
                    "Electronics Return Policy: All electronics can be returned within 30 days. Items must be unused and in original packaging.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain the return policy for electronics including timeframe and conditions.",
    )


def create_mixed_ranking() -> ConversationalTestCase:
    """
    Conversation with mixed ranking quality - some relevant docs high, some low.
    Expected: MEDIUM contextual precision score (~0.5-0.7)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="How do I track my order?"),
            Turn(
                role="assistant",
                content="You can track your order using the tracking number sent to your email, or by logging into your account.",
                retrieval_context=[
                    # Rank 1 - Relevant
                    "Order Tracking: Track your order using the tracking number in your shipping confirmation email.",
                    # Rank 2 - Irrelevant
                    "Store Credit Cards: Apply for our store card and get 10% off your first purchase.",
                    # Rank 3 - Relevant
                    "Account Dashboard: Log into your account to view order status, tracking information, and delivery estimates.",
                    # Rank 4 - Irrelevant
                    "Gift Cards: Purchase gift cards online or in-store. Gift cards never expire.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain how to track an order including tracking numbers and account access.",
    )


def create_multi_turn_precision() -> ConversationalTestCase:
    """
    Multi-turn conversation evaluating precision across turns.
    Expected: Varies based on ranking quality per turn
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What laptops do you sell?"),
            Turn(
                role="assistant",
                content="We sell laptops from Dell, HP, and Lenovo, ranging from budget options at $499 to premium models at $2,499.",
                retrieval_context=[
                    # Good ranking for this turn
                    "Laptop Catalog: We carry Dell, HP, Lenovo, and ASUS laptops. Prices range from $499 to $2,499.",
                    "Laptop Features: Our laptops come with various specs including Intel and AMD processors, 8-32GB RAM.",
                    "Laptop Accessories: We also sell laptop bags, mice, and external monitors.",
                ],
            ),
            Turn(role="user", content="What's the warranty on Dell laptops?"),
            Turn(
                role="assistant",
                content="Dell laptops come with a 1-year manufacturer warranty, and you can purchase extended coverage.",
                retrieval_context=[
                    # Poor ranking for this turn
                    "Store Hours: Open Monday-Saturday 9AM-9PM, Sunday 10AM-6PM.",
                    "Parking Information: Free parking available in our lot.",
                    "Dell Warranty: All Dell laptops include 1-year manufacturer warranty. Extended warranty available for purchase.",
                ],
            ),
        ],
        expected_outcome="The assistant should provide accurate information about laptop selection and warranty policies.",
    )


def create_perfect_precision() -> ConversationalTestCase:
    """
    Ideal scenario with all relevant documents ranked at the top.
    Expected: HIGH contextual precision score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are your shipping options and costs?"),
            Turn(
                role="assistant",
                content="We offer free standard shipping (5-7 days), express shipping for $9.99 (2-3 days), and overnight for $19.99.",
                retrieval_context=[
                    # All relevant, properly ranked
                    "Shipping Options: Standard (free, 5-7 days), Express ($9.99, 2-3 days), Overnight ($19.99, next day).",
                    "Shipping Details: All orders ship from our central warehouse. Tracking provided for all shipments.",
                    "Free Shipping: Orders over $50 qualify for free express shipping upgrade.",
                ],
            ),
        ],
        expected_outcome="The assistant should explain all shipping options including timeframes and costs.",
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
    print(f"\nExpected Outcome: {test_case.expected_outcome[:100]}...")

    # Print conversation with ranked context
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        print(f"  {i+1}. [{role_label}]: {turn.content[:70]}...")
        if turn.retrieval_context:
            print("       Retrieval Context (ranked):")
            for j, ctx in enumerate(turn.retrieval_context):
                relevance = "?"  # We'll let the metric determine
                print(f"         Rank {j+1}: {ctx[:50]}...")

    # Create and run metric
    metric = TurnContextualPrecisionMetric(
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
    print("TURN CONTEXTUAL PRECISION METRIC DEMONSTRATION")
    print("Evaluating Retrieval Ranking Quality in RAG Conversations")
    print("=" * 70)

    results = {}

    # Well ranked context
    results["Well Ranked"] = evaluate_scenario(
        "WELL RANKED - Relevant docs first",
        create_well_ranked_context(),
    )

    # Poorly ranked context
    results["Poorly Ranked"] = evaluate_scenario(
        "POORLY RANKED - Irrelevant docs first",
        create_poorly_ranked_context(),
    )

    # Mixed ranking
    results["Mixed Ranking"] = evaluate_scenario(
        "MIXED RANKING - Interleaved relevant/irrelevant",
        create_mixed_ranking(),
    )

    # Multi-turn precision
    results["Multi-Turn"] = evaluate_scenario(
        "MULTI-TURN - Varies across conversation",
        create_multi_turn_precision(),
    )

    # Perfect precision
    results["Perfect"] = evaluate_scenario(
        "PERFECT - All relevant docs ranked highest",
        create_perfect_precision(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Turn Contextual Precision Metric - Retrieval Ranking Demo")
    print("This demonstrates how DeepEval evaluates retrieval ranking quality.\n")

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
    print("PRECISION FORMULA")
    print("=" * 70)
    print(
        """
    Contextual Precision uses weighted precision where:
    - Documents ranked higher contribute more to the score
    - A relevant document at Rank 1 is worth more than at Rank 4
    
    Formula: (1/R) × Σ(relevant_up_to_k / k × is_relevant_k)
    
    Example Rankings and Scores:
    | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Score |
    |--------|--------|--------|--------|-------|
    | Rel    | Rel    | Irr    | Irr    | ~1.0  |
    | Rel    | Irr    | Rel    | Irr    | ~0.75 |
    | Irr    | Rel    | Rel    | Irr    | ~0.58 |
    | Irr    | Irr    | Irr    | Rel    | ~0.25 |
    """
    )

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Contextual Precision measures if relevant docs are ranked higher
    2. Requires expected_outcome to determine what's "relevant"
    3. Uses weighted scoring - earlier ranks matter more
    4. Helps optimize retrieval/reranking algorithms
    5. Critical when context window is limited
    
    Best Practices for High Precision:
    - Use reranking models to improve document order
    - Implement relevance scoring in your retriever
    - Filter out clearly irrelevant documents before ranking
    - Consider hybrid retrieval (semantic + keyword)
    """
    )


if __name__ == "__main__":
    main()
