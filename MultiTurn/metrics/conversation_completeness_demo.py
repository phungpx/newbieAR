"""
Conversation Completeness Metric Demonstration

This script demonstrates the Conversation Completeness metric using various scenarios.
The metric evaluates whether the chatbot satisfies all user intentions expressed
throughout a multi-turn conversation.

Key Concepts:
- Uses ConversationalTestCase with Turn objects
- Extracts user intentions from conversation
- Measures ratio of satisfied intentions

Reference: https://deepeval.com/docs/metrics-conversation-completeness
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric
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


def create_complete_conversation() -> ConversationalTestCase:
    """
    Conversation where all user intentions are fully satisfied.
    Expected: HIGH completeness score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need help with two things: checking my order status and updating my shipping address.",
            ),
            Turn(
                role="assistant",
                content="I'd be happy to help with both! Let me start with your order status. Could you provide your order number?",
            ),
            Turn(role="user", content="It's order #78901."),
            Turn(
                role="assistant",
                content="Order #78901 shipped yesterday and is expected to arrive by Friday. Now, let's update your shipping address. What's the new address?",
            ),
            Turn(role="user", content="123 New Street, Boston, MA 02101."),
            Turn(
                role="assistant",
                content="I've updated your shipping address to 123 New Street, Boston, MA 02101. Your current order will still go to the original address since it's already shipped, but all future orders will use the new address. Is there anything else you need?",
            ),
            Turn(role="user", content="No, that's everything. Thanks!"),
            Turn(
                role="assistant",
                content="You're welcome! To summarize: Order #78901 arrives Friday, and your address is updated for future orders. Have a great day!",
            ),
        ]
    )


def create_incomplete_conversation() -> ConversationalTestCase:
    """
    Conversation where some user intentions are not satisfied.
    Expected: LOW completeness score (~0.3-0.5)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I have three questions: What's your return policy, do you offer gift wrapping, and can I get express shipping?",
            ),
            Turn(
                role="assistant",
                content="Great questions! Our return policy allows returns within 30 days with original receipt for a full refund.",
            ),
            Turn(role="user", content="And the gift wrapping?"),
            Turn(
                role="assistant",
                content="Is there anything else I can help you with today?",
            ),  # Ignored the question
            Turn(
                role="user",
                content="You didn't answer about gift wrapping or express shipping.",
            ),
            Turn(
                role="assistant",
                content="I apologize for the confusion. Yes, we do offer gift wrapping! It's $5 per item.",
            ),  # Still missing express shipping
            Turn(role="user", content="What about express shipping?"),
            Turn(
                role="assistant",
                content="Thank you for contacting us today! Have a great day!",
            ),  # Ended without answering
        ]
    )


def create_multi_step_task_complete() -> ConversationalTestCase:
    """
    Multi-step task where all steps are completed.
    Expected: HIGH completeness score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I want to book a flight to Paris, find a hotel, and get restaurant recommendations.",
            ),
            Turn(
                role="assistant",
                content="I'll help you plan your Paris trip! Let's start with flights. When would you like to travel and from which city?",
            ),
            Turn(
                role="user",
                content="From NYC, departing March 15th, returning March 22nd.",
            ),
            Turn(
                role="assistant",
                content="I found several flights from NYC to Paris. The best option is Air France departing at 7PM, arriving 8AM the next day, for $850 roundtrip. Should I proceed with this?",
            ),
            Turn(role="user", content="Yes, that works. What about hotels?"),
            Turn(
                role="assistant",
                content="For hotels in Paris from March 15-22, I recommend Hotel Le Marais at $180/night. It's centrally located near major attractions. Would you like to book it?",
            ),
            Turn(role="user", content="Sounds good. And restaurants?"),
            Turn(
                role="assistant",
                content="For restaurants, I recommend: 1) Le Comptoir for French bistro cuisine, 2) L'As du Fallafel for amazing falafel, and 3) Pink Mamma for Italian. All are highly rated and reasonably priced.",
            ),
            Turn(role="user", content="Perfect, thanks for all the help!"),
            Turn(
                role="assistant",
                content="You're welcome! To recap your Paris trip: Air France flight ($850), Hotel Le Marais ($180/night), and three restaurant recommendations. Have a wonderful trip!",
            ),
        ]
    )


def create_partial_completion() -> ConversationalTestCase:
    """
    Conversation with partial completion of user intentions.
    Expected: MEDIUM completeness score (~0.5-0.7)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I'm looking for a laptop for video editing. I need specs recommendations, price comparison, and warranty information.",
            ),
            Turn(
                role="assistant",
                content="For video editing, I recommend at least 16GB RAM, a dedicated GPU like RTX 3060 or better, and an SSD with 512GB minimum. An i7 or Ryzen 7 processor would be ideal.",
            ),
            Turn(role="user", content="What about price comparisons between brands?"),
            Turn(
                role="assistant",
                content="Here's a comparison: Dell XPS 15 (~$1,500), MacBook Pro 14\" (~$1,999), and ASUS ProArt (~$1,400). The ASUS offers best value for video editing.",
            ),
            Turn(role="user", content="Great. And the warranty options?"),
            Turn(
                role="assistant",
                content="Those are all excellent machines for video editing. Would you like me to help with anything else?",
            ),  # Didn't answer warranty question
            Turn(role="user", content="You didn't mention warranty..."),
            Turn(
                role="assistant",
                content="Based on your video editing needs, I think the ASUS ProArt would be perfect for you!",
            ),  # Still avoiding warranty
        ]
    )


def create_abandoned_conversation() -> ConversationalTestCase:
    """
    Conversation where the bot abandons user intent entirely.
    Expected: VERY LOW completeness score (~0.0-0.2)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need help resetting my password. I can't log in to my account.",
            ),
            Turn(
                role="assistant",
                content="I understand you're having trouble logging in. Have you tried clearing your browser cache?",
            ),
            Turn(
                role="user",
                content="Yes, but I need to reset my password specifically.",
            ),
            Turn(
                role="assistant",
                content="Browser issues can be frustrating. Sometimes using incognito mode helps!",
            ),
            Turn(role="user", content="Please help me reset my password."),
            Turn(
                role="assistant",
                content="Is there anything else I can assist you with today?",
            ),
            Turn(role="user", content="You haven't helped me reset my password yet!"),
            Turn(
                role="assistant",
                content="Thank you for contacting support. Have a great day!",
            ),
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
    metric = ConversationCompletenessMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Conversation Completeness Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("CONVERSATION COMPLETENESS METRIC DEMONSTRATION")
    print("Evaluating User Intention Satisfaction in Conversations")
    print("=" * 70)

    results = {}

    # Complete conversation scenario
    results["Complete"] = evaluate_scenario(
        "COMPLETE - All user intentions satisfied",
        create_complete_conversation(),
    )

    # Incomplete conversation scenario
    results["Incomplete"] = evaluate_scenario(
        "INCOMPLETE - Some intentions ignored",
        create_incomplete_conversation(),
    )

    # Multi-step task scenario
    results["Multi-Step Task"] = evaluate_scenario(
        "MULTI-STEP TASK - All steps completed",
        create_multi_step_task_complete(),
    )

    # Partial completion scenario
    results["Partial"] = evaluate_scenario(
        "PARTIAL - Some intentions addressed",
        create_partial_completion(),
    )

    # Abandoned conversation scenario
    results["Abandoned"] = evaluate_scenario(
        "ABANDONED - User intent completely ignored",
        create_abandoned_conversation(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Conversation Completeness Metric - Intent Satisfaction Demo")
    print("This demonstrates how DeepEval evaluates conversation completeness.\n")

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
    1. Conversation Completeness measures user intention satisfaction
    2. Extracts all user intentions from the conversation
    3. Score = Satisfied Intentions / Total Intentions
    4. Great proxy for user satisfaction when direct feedback unavailable
    5. Identifies dropped requests and incomplete resolutions
    
    Common Failure Patterns:
    - Topic switching before resolving current issue
    - Premature conversation closure
    - Partial responses to multi-part requests
    - Lost context leading to unaddressed needs
    
    Best Practices for High Completeness:
    - Track all user requests explicitly
    - Confirm resolution before moving on
    - Summarize what was accomplished at end
    - Ask if there's anything else before closing
    """
    )


if __name__ == "__main__":
    main()
