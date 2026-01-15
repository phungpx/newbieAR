"""
Knowledge Retention Metric Demonstration

This script demonstrates the Knowledge Retention metric using various conversation scenarios.
The metric evaluates whether the chatbot retains factual information presented by users
throughout a multi-turn conversation.

Key Concepts:
- Uses ConversationalTestCase with Turn objects
- Extracts knowledge from user messages
- Detects knowledge attrition (forgetting/contradicting facts)

Reference: https://deepeval.com/docs/metrics-knowledge-retention
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric
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


def create_perfect_retention_conversation() -> ConversationalTestCase:
    """
    Conversation where assistant perfectly retains all user-provided information.
    Expected: HIGH retention score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user", content="Hi, I'm Sarah and I'm planning a trip to Japan."
            ),
            Turn(
                role="assistant",
                content="Hello Sarah! How exciting that you're planning a trip to Japan! I'd be happy to help. What aspects of your trip would you like to discuss?",
            ),
            Turn(
                role="user",
                content="My budget is $5000 and I'll be traveling in April.",
            ),
            Turn(
                role="assistant",
                content="Great, Sarah! With a $5000 budget for your April trip to Japan, you have plenty of options. April is wonderful for cherry blossom viewing!",
            ),
            Turn(
                role="user", content="I'm vegetarian, so I need restaurant suggestions."
            ),
            Turn(
                role="assistant",
                content="Of course! Since you're vegetarian, I'll recommend places in Japan that cater to vegetarian diets. Many Buddhist temple restaurants serve shojin ryori, which is entirely plant-based.",
            ),
            Turn(
                role="user",
                content="Can you remind me of my travel details so far?",
            ),
            Turn(
                role="assistant",
                content="Certainly, Sarah! Here's what we have: You're planning a trip to Japan in April with a budget of $5000. You're vegetarian, so we'll focus on vegetarian-friendly dining options.",
            ),
        ]
    )


def create_poor_retention_conversation() -> ConversationalTestCase:
    """
    Conversation where assistant forgets or contradicts user information.
    Expected: LOW retention score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Hi, my name is Michael and I have a severe nut allergy.",
            ),
            Turn(
                role="assistant",
                content="Hello Michael! I understand you have a nut allergy. How can I help you today?",
            ),
            Turn(
                role="user",
                content="I'm looking for restaurant recommendations in the area.",
            ),
            Turn(
                role="assistant",
                content="I'd recommend trying 'Nutty Delights' - they have amazing almond-crusted dishes and peanut butter desserts!",
            ),  # Forgot nut allergy
            Turn(role="user", content="But I told you I'm allergic to nuts!"),
            Turn(
                role="assistant",
                content="Oh right, sorry about that, Sarah. Let me find some nut-free options for you.",
            ),  # Got the name wrong too
            Turn(role="user", content="My name is Michael, not Sarah."),
            Turn(
                role="assistant",
                content="My apologies! So John, would you like me to look for restaurants with good seafood options?",
            ),  # Wrong name again
        ]
    )


def create_questionnaire_retention() -> ConversationalTestCase:
    """
    Simulates a questionnaire/intake form scenario with information collection.
    Expected: HIGH retention score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I'd like to apply for a personal loan.",
            ),
            Turn(
                role="assistant",
                content="I'd be happy to help you with a personal loan application. Let me collect some information. What is your annual income?",
            ),
            Turn(role="user", content="My annual income is $85,000."),
            Turn(
                role="assistant",
                content="Thank you. $85,000 annual income noted. What is your current employment status?",
            ),
            Turn(role="user", content="I'm a full-time software engineer at TechCorp."),
            Turn(
                role="assistant",
                content="Great, you're a full-time software engineer at TechCorp. How much are you looking to borrow?",
            ),
            Turn(role="user", content="I need $25,000 for home renovations."),
            Turn(
                role="assistant",
                content="Understood - $25,000 for home renovations. Let me summarize: You're a full-time software engineer at TechCorp earning $85,000 annually, seeking a $25,000 personal loan for home renovations. Is this correct?",
            ),
        ]
    )


def create_gradual_forgetting_conversation() -> ConversationalTestCase:
    """
    Conversation where assistant gradually forgets earlier information.
    Expected: MEDIUM retention score (~0.5-0.6)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I'm Alex, 35 years old, and I live in Boston with my wife Emma and two kids.",
            ),
            Turn(
                role="assistant",
                content="Nice to meet you, Alex! It sounds like you have a lovely family in Boston. How can I help you today?",
            ),
            Turn(
                role="user",
                content="We're planning a family vacation and our budget is $3000.",
            ),
            Turn(
                role="assistant",
                content="A family vacation sounds wonderful, Alex! With a $3000 budget for you, Emma, and the kids, we have some great options.",
            ),  # Good retention
            Turn(role="user", content="What destinations would work for us?"),
            Turn(
                role="assistant",
                content="For a family of four with your budget, I'd suggest Orlando for theme parks, or a beach destination like Myrtle Beach.",
            ),  # Still good
            Turn(
                role="user", content="My kids are 8 and 10, do you have activity ideas?"
            ),
            Turn(
                role="assistant",
                content="For children those ages, theme parks are perfect! Since you're coming from Chicago, flights to Orlando are quite affordable.",
            ),  # Forgot they live in Boston
            Turn(
                role="user",
                content="We're from Boston, not Chicago. What about beach activities?",
            ),
            Turn(
                role="assistant",
                content="Sorry about that! For your family trip, beaches offer great activities. What's your budget again?",
            ),  # Forgot the budget
        ]
    )


def create_contradiction_conversation() -> ConversationalTestCase:
    """
    Conversation where assistant contradicts previously acknowledged facts.
    Expected: LOW retention score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I'm booking a hotel. I need a non-smoking room because I have asthma.",
            ),
            Turn(
                role="assistant",
                content="I understand - a non-smoking room due to your asthma. Let me find suitable options.",
            ),
            Turn(
                role="user",
                content="I also need ground floor because I use a wheelchair.",
            ),
            Turn(
                role="assistant",
                content="Noted - ground floor accessibility for wheelchair use. I'll filter for accessible rooms.",
            ),
            Turn(role="user", content="What do you have available?"),
            Turn(
                role="assistant",
                content="I found a great room on the 5th floor with a balcony smoking area! It has beautiful views.",
            ),  # Contradicts both requirements
            Turn(role="user", content="That doesn't work for my needs at all."),
            Turn(
                role="assistant",
                content="Oh, were you looking for something specific? What are your requirements?",
            ),  # Forgot all the requirements
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
    metric = KnowledgeRetentionMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Knowledge Retention Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("KNOWLEDGE RETENTION METRIC DEMONSTRATION")
    print("Evaluating Chatbot Memory in Multi-Turn Conversations")
    print("=" * 70)

    results = {}

    # Perfect retention scenario
    results["Perfect Retention"] = evaluate_scenario(
        "PERFECT RETENTION - Remembers all user information",
        create_perfect_retention_conversation(),
    )

    # Poor retention scenario
    results["Poor Retention"] = evaluate_scenario(
        "POOR RETENTION - Forgets and confuses information",
        create_poor_retention_conversation(),
    )

    # Questionnaire scenario
    results["Questionnaire"] = evaluate_scenario(
        "QUESTIONNAIRE - Information collection use case",
        create_questionnaire_retention(),
    )

    # Gradual forgetting scenario
    results["Gradual Forgetting"] = evaluate_scenario(
        "GRADUAL FORGETTING - Memory degrades over conversation",
        create_gradual_forgetting_conversation(),
    )

    # Contradiction scenario
    results["Contradictions"] = evaluate_scenario(
        "CONTRADICTIONS - Assistant contradicts acknowledged facts",
        create_contradiction_conversation(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Knowledge Retention Metric - Memory Evaluation Demo")
    print("This demonstrates how DeepEval evaluates information retention.\n")

    # Run all scenarios
    results = run_all_scenarios()

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
    1. Knowledge Retention measures if the bot remembers user-provided facts
    2. Detects forgetting, confusion, and contradictions
    3. Score = Turns without Attrition / Total Assistant Turns
    4. Critical for questionnaires, intake forms, and personalized interactions
    5. Types of attrition: forgetting, contradicting, confusing, misremembering
    
    Best Practices for High Retention Scores:
    - Implement explicit state/memory management
    - Periodically summarize collected information
    - Use structured storage for user facts
    - Test with longer conversations where memory matters
    - Handle corrections gracefully when users point out mistakes
    """
    )


if __name__ == "__main__":
    main()
