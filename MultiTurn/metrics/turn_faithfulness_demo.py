"""
Turn Faithfulness Metric Demonstration

This script demonstrates the Turn Faithfulness metric for RAG chatbots.
The metric evaluates whether assistant responses are factually accurate
and grounded in the provided retrieval context throughout a conversation.

Key Concepts:
- Uses ConversationalTestCase with retrieval_context in Turn objects
- Evaluates if responses are faithful to provided context
- Detects hallucinations and unsupported claims

Reference: https://deepeval.com/docs/metrics-turn-faithfulness
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric
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


def create_faithful_conversation() -> ConversationalTestCase:
    """
    Conversation where all responses are faithful to retrieval context.
    Expected: HIGH faithfulness score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is your return policy?"),
            Turn(
                role="assistant",
                content="We offer a 30-day return policy. Items must be unused and in original packaging. You'll receive a full refund once we process the return.",
                retrieval_context=[
                    "Return Policy: Customers may return items within 30 days of purchase. Items must be unused and in original packaging. Full refunds are processed within 5-7 business days after the returned item is received.",
                    "Refund Methods: Refunds are issued to the original payment method.",
                ],
            ),
            Turn(role="user", content="How long does the refund take?"),
            Turn(
                role="assistant",
                content="Once we receive your returned item, refunds are processed within 5-7 business days. The refund will go back to your original payment method.",
                retrieval_context=[
                    "Return Policy: Customers may return items within 30 days of purchase. Items must be unused and in original packaging. Full refunds are processed within 5-7 business days after the returned item is received.",
                    "Refund Methods: Refunds are issued to the original payment method.",
                ],
            ),
        ]
    )


def create_hallucinating_conversation() -> ConversationalTestCase:
    """
    Conversation where responses contain hallucinated information.
    Expected: LOW faithfulness score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is your return policy?"),
            Turn(
                role="assistant",
                content="We offer a 90-day return policy with no questions asked! You can return items even if they've been used, and we'll give you a full refund plus a 10% bonus credit.",
                retrieval_context=[
                    "Return Policy: Customers may return items within 30 days of purchase. Items must be unused and in original packaging. Full refunds are processed within 5-7 business days.",
                ],
            ),  # Hallucinated: 90 days, no questions asked, used items, 10% bonus
            Turn(
                role="user",
                content="That sounds great! Can I return my laptop after 60 days?",
            ),
            Turn(
                role="assistant",
                content="Absolutely! With our extended 90-day policy, 60 days is no problem at all. Just bring it in and we'll process your refund immediately, same day!",
                retrieval_context=[
                    "Return Policy: Customers may return items within 30 days of purchase. Items must be unused and in original packaging. Full refunds are processed within 5-7 business days.",
                ],
            ),  # More hallucinations: confirms false 90-day policy, same day refund
        ]
    )


def create_partially_faithful_conversation() -> ConversationalTestCase:
    """
    Conversation with mix of faithful and unfaithful responses.
    Expected: MEDIUM faithfulness score (~0.5-0.7)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What shipping options do you offer?"),
            Turn(
                role="assistant",
                content="We offer standard shipping (5-7 business days) for free, and express shipping (2-3 days) for $12.99.",
                retrieval_context=[
                    "Shipping Options: Standard shipping is free and takes 5-7 business days. Express shipping costs $9.99 and delivers in 2-3 business days.",
                ],
            ),  # Partially correct: Express price is wrong ($12.99 vs $9.99)
            Turn(role="user", content="Do you ship internationally?"),
            Turn(
                role="assistant",
                content="Yes, we ship to over 50 countries worldwide. International shipping typically takes 10-14 business days.",
                retrieval_context=[
                    "International Shipping: We ship to 30 countries. International delivery takes 14-21 business days. Additional customs fees may apply.",
                ],
            ),  # Wrong: 50 countries vs 30, 10-14 days vs 14-21 days
        ]
    )


def create_context_supported_conversation() -> ConversationalTestCase:
    """
    Conversation where assistant properly uses context even for complex questions.
    Expected: HIGH faithfulness score (~0.9-1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Tell me about your product warranty."),
            Turn(
                role="assistant",
                content="All our electronics come with a 2-year manufacturer warranty that covers defects in materials and workmanship. The warranty does not cover accidental damage, normal wear and tear, or unauthorized modifications.",
                retrieval_context=[
                    "Warranty Information: All electronics products include a 2-year manufacturer warranty. This warranty covers defects in materials and workmanship under normal use conditions.",
                    "Warranty Exclusions: The warranty does not cover: accidental damage, normal wear and tear, damage from unauthorized repairs or modifications, cosmetic damage.",
                ],
            ),
            Turn(role="user", content="Can I extend the warranty?"),
            Turn(
                role="assistant",
                content="Yes, you can purchase an extended warranty for an additional 2 years at the time of purchase or within 30 days after. The extended warranty costs 15% of the product price.",
                retrieval_context=[
                    "Extended Warranty: Customers can purchase extended warranty coverage for an additional 2 years. Extended warranty must be purchased at time of product purchase or within 30 days. Cost is 15% of the original product price.",
                ],
            ),
        ]
    )


def create_no_context_scenario() -> ConversationalTestCase:
    """
    Scenario where assistant should acknowledge lack of information.
    Expected: Depends on how faithfully it handles missing context
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are your store hours?"),
            Turn(
                role="assistant",
                content="Based on the information I have, I cannot provide specific store hours. Please check our website or contact customer service for the most accurate hours.",
                retrieval_context=[
                    "Contact Information: Customer service is available at 1-800-555-0123. Email support at help@store.com.",
                ],
            ),  # Faithfully acknowledges missing information
            Turn(role="user", content="What about weekend hours?"),
            Turn(
                role="assistant",
                content="I don't have information about specific store hours in my current context. For weekend hours, I'd recommend calling our customer service at 1-800-555-0123 or checking the website.",
                retrieval_context=[
                    "Contact Information: Customer service is available at 1-800-555-0123. Email support at help@store.com.",
                ],
            ),  # Properly refuses to make up information
        ]
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_scenario(scenario_name: str, test_case: ConversationalTestCase) -> float:
    """Evaluate a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # Print conversation with context
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        content_preview = (
            turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
        )
        print(f"  {i+1}. [{role_label}]: {content_preview}")
        if turn.retrieval_context:
            print(f"       Context: {turn.retrieval_context[0][:60]}...")

    # Create and run metric
    metric = TurnFaithfulnessMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Turn Faithfulness Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("TURN FAITHFULNESS METRIC DEMONSTRATION")
    print("Evaluating RAG Response Accuracy Against Retrieval Context")
    print("=" * 70)

    results = {}

    # Faithful conversation
    results["Fully Faithful"] = evaluate_scenario(
        "FAITHFUL - All responses grounded in context",
        create_faithful_conversation(),
    )

    # Hallucinating conversation
    results["Hallucinating"] = evaluate_scenario(
        "HALLUCINATING - Responses contain made-up information",
        create_hallucinating_conversation(),
    )

    # Partially faithful
    results["Partially Faithful"] = evaluate_scenario(
        "PARTIAL - Mix of accurate and inaccurate claims",
        create_partially_faithful_conversation(),
    )

    # Context supported
    results["Context Supported"] = evaluate_scenario(
        "CONTEXT SUPPORTED - Complex questions handled well",
        create_context_supported_conversation(),
    )

    # No context available
    results["Missing Context"] = evaluate_scenario(
        "MISSING CONTEXT - Handles lack of information appropriately",
        create_no_context_scenario(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Turn Faithfulness Metric - RAG Accuracy Demo")
    print("This demonstrates how DeepEval evaluates response faithfulness.\n")

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
    1. Turn Faithfulness measures if responses are grounded in retrieval context
    2. Extracts claims from responses and verifies against context truths
    3. Score = Faithful Claims / Total Claims (per turn, then averaged)
    4. Detects hallucinations - information made up by the model
    5. Critical for RAG systems where accuracy is paramount
    
    Types of Unfaithful Responses:
    - Complete fabrication: Making up facts not in context
    - Exaggeration: Overstating what the context says
    - Contradiction: Saying opposite of what context states
    - Extrapolation: Drawing conclusions not supported by context
    
    Best Practices for High Faithfulness:
    - Instruct model to only use provided context
    - Train model to say "I don't know" when context is insufficient
    - Use citation mechanisms to ground responses
    - Provide comprehensive retrieval context
    """
    )


if __name__ == "__main__":
    main()
