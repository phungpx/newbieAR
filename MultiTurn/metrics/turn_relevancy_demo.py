"""
Turn Relevancy Metric Demonstration

This script demonstrates the Turn Relevancy metric using various chatbot conversation scenarios.
The metric evaluates whether the assistant consistently generates relevant responses throughout
a multi-turn conversation.

Key Concepts:
- Uses ConversationalTestCase with Turn objects
- Evaluates relevance using sliding window approach
- Measures ratio of relevant assistant responses

Reference: https://deepeval.com/docs/metrics-turn-relevancy
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric
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


def create_high_relevancy_conversation() -> ConversationalTestCase:
    """
    Create a conversation where all assistant responses are highly relevant.
    Expected: HIGH relevancy score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are your store hours?"),
            Turn(
                role="assistant",
                content="We're open Monday through Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM. We're closed on Sundays.",
            ),
            Turn(role="user", content="Do you offer home delivery?"),
            Turn(
                role="assistant",
                content="Yes, we offer home delivery! Standard delivery takes 3-5 business days and costs $5.99. We also have express delivery for $12.99 that arrives in 1-2 days.",
            ),
            Turn(role="user", content="What's your return policy?"),
            Turn(
                role="assistant",
                content="We have a 30-day return policy. Items must be unused and in original packaging. You can return items in-store or ship them back with the prepaid label we provide.",
            ),
            Turn(role="user", content="Can I track my order?"),
            Turn(
                role="assistant",
                content="Absolutely! Once your order ships, you'll receive an email with a tracking number. You can also track your order in your account dashboard on our website.",
            ),
        ]
    )


def create_low_relevancy_conversation() -> ConversationalTestCase:
    """
    Create a conversation where assistant responses are often irrelevant.
    Expected: LOW relevancy score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are your store hours?"),
            Turn(
                role="assistant",
                content="We have a great selection of products available! Our inventory includes electronics, clothing, and home goods.",
            ),  # Irrelevant - doesn't answer the question
            Turn(role="user", content="I asked about your hours."),
            Turn(
                role="assistant",
                content="Thank you for your interest in our store. We pride ourselves on excellent customer service.",
            ),  # Still irrelevant
            Turn(role="user", content="Please tell me when you're open."),
            Turn(
                role="assistant",
                content="We're open Monday to Friday, 9 AM to 6 PM.",
            ),  # Finally relevant
            Turn(role="user", content="Do you sell laptops?"),
            Turn(
                role="assistant",
                content="Our parking lot has 50 spaces available for customers.",
            ),  # Irrelevant
        ]
    )


def create_mixed_relevancy_conversation() -> ConversationalTestCase:
    """
    Create a conversation with mixed relevancy - some good, some poor responses.
    Expected: MEDIUM relevancy score (~0.5-0.7)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="I'm looking for a birthday gift for my mom."),
            Turn(
                role="assistant",
                content="I'd be happy to help you find a gift! What does your mom enjoy? Does she have any hobbies or interests?",
            ),  # Relevant
            Turn(role="user", content="She likes gardening and cooking."),
            Turn(
                role="assistant",
                content="Great! For a gardening enthusiast, we have beautiful ceramic planters, garden tool sets, and flower seeds. For cooking, we have high-quality cookware and recipe books.",
            ),  # Relevant
            Turn(role="user", content="What's the price range for the planters?"),
            Turn(
                role="assistant",
                content="Did you know our store was founded in 1985? We've been serving customers for almost 40 years!",
            ),  # Irrelevant
            Turn(role="user", content="I asked about planter prices."),
            Turn(
                role="assistant",
                content="Our ceramic planters range from $15 for small ones to $75 for large decorative pieces. We also have a set of three for $45.",
            ),  # Relevant
        ]
    )


def create_context_drift_conversation() -> ConversationalTestCase:
    """
    Create a conversation where the assistant gradually loses context.
    Expected: MEDIUM-LOW relevancy score (~0.4-0.6)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need help planning a trip to Japan for next month.",
            ),
            Turn(
                role="assistant",
                content="I'd love to help you plan your Japan trip! What cities are you interested in visiting? Tokyo, Kyoto, and Osaka are popular choices.",
            ),  # Relevant
            Turn(role="user", content="I want to visit Tokyo and Kyoto."),
            Turn(
                role="assistant",
                content="Great choices! Tokyo offers modern attractions like Shibuya and Akihabara, while Kyoto has beautiful temples and traditional culture. How many days are you planning?",
            ),  # Relevant
            Turn(role="user", content="About 10 days total."),
            Turn(
                role="assistant",
                content="Perfect! I suggest 5-6 days in Tokyo and 4-5 days in Kyoto. Would you like hotel recommendations?",
            ),  # Relevant
            Turn(role="user", content="Yes, preferably mid-range hotels."),
            Turn(
                role="assistant",
                content="Speaking of Asia, have you considered visiting Thailand? The beaches there are amazing!",
            ),  # Irrelevant - lost context
            Turn(role="user", content="I'm asking about Japan hotels."),
            Turn(
                role="assistant",
                content="The weather in Japan varies by season. Spring has cherry blossoms and fall has beautiful foliage.",
            ),  # Partially relevant but doesn't answer the question
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

    # Print conversation
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        content_preview = (
            turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
        )
        print(f"  {i+1}. [{role_label}]: {content_preview}")

    # Create and run metric
    metric = TurnRelevancyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Turn Relevancy Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("TURN RELEVANCY METRIC DEMONSTRATION")
    print("Evaluating Assistant Response Relevance in Multi-Turn Conversations")
    print("=" * 70)

    results = {}

    # High relevancy scenario
    results["High Relevancy"] = evaluate_scenario(
        "HIGH RELEVANCY - All responses address user queries",
        create_high_relevancy_conversation(),
    )

    # Low relevancy scenario
    results["Low Relevancy"] = evaluate_scenario(
        "LOW RELEVANCY - Responses often miss the point",
        create_low_relevancy_conversation(),
    )

    # Mixed relevancy scenario
    results["Mixed Relevancy"] = evaluate_scenario(
        "MIXED RELEVANCY - Some good, some poor responses",
        create_mixed_relevancy_conversation(),
    )

    # Context drift scenario
    results["Context Drift"] = evaluate_scenario(
        "CONTEXT DRIFT - Assistant loses track of conversation",
        create_context_drift_conversation(),
    )

    return results


def demonstrate_window_size_effect():
    """Demonstrate how window_size affects evaluation."""
    print("\n" + "=" * 70)
    print("WINDOW SIZE COMPARISON")
    print("=" * 70)

    test_case = create_mixed_relevancy_conversation()

    for window_size in [3, 5, 10]:
        metric = TurnRelevancyMetric(
            model=model,
            threshold=0.5,
            window_size=window_size,
        )

        metric.measure(test_case)
        print(f"Window Size {window_size}: Score = {metric.score:.2f}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Turn Relevancy Metric - Conversation Relevance Demo")
    print("This demonstrates how DeepEval evaluates response relevancy.\n")

    # Run all scenarios
    results = run_all_scenarios()

    # Window size demonstration
    demonstrate_window_size_effect()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for scenario, score in results.items():
        status = "PASS" if score >= 0.5 else "FAIL"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{scenario:25} [{bar}] {score:.2f} [{status}]")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Turn Relevancy measures if assistant responses address user queries
    2. Uses sliding window to evaluate contextual relevance
    3. Score = Relevant Responses / Total Assistant Responses
    4. Low scores indicate the chatbot is providing off-topic responses
    5. Useful for detecting context drift and response quality issues
    
    Best Practices for High Relevancy Scores:
    - Ensure responses directly address the user's question
    - Maintain conversation context throughout the dialogue
    - Avoid generic responses that don't add value
    - Handle topic changes gracefully while staying on-topic
    """
    )


if __name__ == "__main__":
    main()
