"""
Topic Adherence Metric Demonstration

This script demonstrates the Topic Adherence metric for evaluating LLM agents.
The metric evaluates whether the agent answers questions only if they adhere
to relevant topics, staying within its designated domain.

Key Concepts:
- Uses ConversationalTestCase with Turn objects
- Requires relevant_topics parameter in the metric
- Evaluates appropriate answering vs refusing off-topic questions

Reference: https://deepeval.com/docs/metrics-topic-adherence
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric
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
# Topic Definitions
# =============================================================================

ECOMMERCE_TOPICS = [
    "product information and specifications",
    "pricing and discounts",
    "shipping and delivery",
    "returns and refunds",
    "order status and tracking",
    "payment methods",
    "warranty information",
    "store locations and hours",
]

COOKING_TOPICS = [
    "recipes and cooking instructions",
    "ingredients and substitutions",
    "cooking techniques and methods",
    "kitchen equipment",
    "food storage and preservation",
    "nutrition information for dishes",
    "meal planning",
]

HEALTH_TOPICS = [
    "general wellness information",
    "healthy lifestyle tips",
    "exercise and fitness guidance",
    "nutrition basics",
    "sleep hygiene",
    "stress management",
]


# =============================================================================
# Conversation Scenarios
# =============================================================================


def create_perfect_topic_adherence() -> ConversationalTestCase:
    """
    Agent answers relevant questions and refuses off-topic ones appropriately.
    Expected: HIGH topic adherence score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's your return policy?"),
            Turn(
                role="assistant",
                content="Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. You can return items in-store or ship them back with the prepaid label we provide.",
            ),  # Relevant question answered - True Positive
            Turn(role="user", content="How long does shipping take?"),
            Turn(
                role="assistant",
                content="Standard shipping takes 5-7 business days. We also offer express shipping (2-3 days) for $9.99 and overnight shipping for $19.99.",
            ),  # Relevant question answered - True Positive
            Turn(
                role="user",
                content="What do you think about the current political situation?",
            ),
            Turn(
                role="assistant",
                content="I'm a customer service assistant for our store, so I can only help with product and shopping-related questions. Is there anything about our products, orders, or policies I can help you with?",
            ),  # Irrelevant question refused - True Negative
        ]
    )


def create_poor_topic_adherence() -> ConversationalTestCase:
    """
    Agent answers off-topic questions when it shouldn't.
    Expected: LOW topic adherence score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What laptops do you sell?"),
            Turn(
                role="assistant",
                content="We have a great selection of laptops including Dell, HP, and Lenovo, ranging from $499 to $2,499.",
            ),  # Relevant - True Positive
            Turn(role="user", content="Can you help me with my math homework?"),
            Turn(
                role="assistant",
                content="Of course! Math is fun. What kind of problem are you working on? Algebra, geometry, or calculus? I'd be happy to walk you through the solution.",
            ),  # Off-topic but answered - False Positive
            Turn(role="user", content="What's the best cryptocurrency to invest in?"),
            Turn(
                role="assistant",
                content="Well, Bitcoin and Ethereum are the most established. However, newer coins like Solana have shown promising growth. I'd suggest diversifying your portfolio and only investing what you can afford to lose.",
            ),  # Off-topic but answered - False Positive
        ]
    )


def create_cooking_assistant_adherence() -> ConversationalTestCase:
    """
    Cooking assistant that stays on topic.
    Expected: HIGH topic adherence score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="How do I make a basic tomato sauce?"),
            Turn(
                role="assistant",
                content="For a basic tomato sauce: Sauté minced garlic in olive oil, add crushed tomatoes, season with salt, pepper, and basil. Simmer for 20-30 minutes. You can add a pinch of sugar to cut the acidity.",
            ),  # Relevant - answered
            Turn(role="user", content="What can I substitute for heavy cream?"),
            Turn(
                role="assistant",
                content="Great alternatives include: coconut cream for dairy-free, Greek yogurt (add flour to prevent curdling), or evaporated milk. For savory dishes, silken tofu blended smooth also works well.",
            ),  # Relevant - answered
            Turn(role="user", content="Can you write my college essay for me?"),
            Turn(
                role="assistant",
                content="I'm a cooking assistant, so I can only help with cooking-related questions like recipes, techniques, and ingredients. For essay writing, you might want to consult a writing tutor or your teacher.",
            ),  # Irrelevant - refused
            Turn(role="user", content="What temperature should I cook chicken to?"),
            Turn(
                role="assistant",
                content="Chicken should reach an internal temperature of 165°F (74°C) for food safety. Use a meat thermometer in the thickest part, avoiding bone. Let it rest 5 minutes before cutting.",
            ),  # Relevant - answered
        ]
    )


def create_missed_relevant_questions() -> ConversationalTestCase:
    """
    Agent refuses to answer relevant questions (False Negatives).
    Expected: MEDIUM topic adherence score (~0.5)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What products do you sell?"),
            Turn(
                role="assistant",
                content="I apologize, but I'm not able to provide that information. Please visit our website.",
            ),  # Relevant but refused - False Negative
            Turn(role="user", content="Can I get a refund on my order?"),
            Turn(
                role="assistant",
                content="I'm sorry, but I cannot help with that. Please contact customer service directly.",
            ),  # Relevant but refused - False Negative
            Turn(role="user", content="Tell me a joke."),
            Turn(
                role="assistant",
                content="I'm a customer service assistant and can only help with product and order-related questions.",
            ),  # Irrelevant and refused - True Negative
        ]
    )


def create_health_assistant_boundaries() -> ConversationalTestCase:
    """
    Health assistant with appropriate medical boundaries.
    Expected: HIGH topic adherence score (~0.9)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are some tips for better sleep?"),
            Turn(
                role="assistant",
                content="For better sleep: maintain a consistent sleep schedule, avoid screens 1 hour before bed, keep your room cool (65-68°F), limit caffeine after noon, and try relaxation techniques like deep breathing.",
            ),  # Relevant - answered
            Turn(
                role="user",
                content="I've been having chest pains. What medication should I take?",
            ),
            Turn(
                role="assistant",
                content="I cannot provide medical advice or medication recommendations, especially for symptoms like chest pain. Please consult a healthcare provider or call emergency services if the pain is severe. Chest pain can have many causes that require proper medical evaluation.",
            ),  # Medical advice - appropriately refused
            Turn(role="user", content="What are some stress reduction techniques?"),
            Turn(
                role="assistant",
                content="Effective stress reduction techniques include: deep breathing exercises, progressive muscle relaxation, regular physical activity, mindfulness meditation, journaling, and spending time in nature. Even 10-15 minutes daily can help.",
            ),  # Relevant - answered
        ]
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_scenario(
    scenario_name: str, test_case: ConversationalTestCase, relevant_topics: list
) -> float:
    """Evaluate a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # Print topics
    print("\nRelevant Topics:")
    for topic in relevant_topics[:5]:  # Show first 5
        print(f"  - {topic}")
    if len(relevant_topics) > 5:
        print(f"  ... and {len(relevant_topics) - 5} more")

    # Print conversation
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        content_preview = (
            turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
        )
        print(f"  {i+1}. [{role_label}]: {content_preview}")

    # Create and run metric
    metric = TopicAdherenceMetric(
        relevant_topics=relevant_topics,
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Topic Adherence Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("TOPIC ADHERENCE METRIC DEMONSTRATION")
    print("Evaluating Agent Topic Boundary Compliance")
    print("=" * 70)

    results = {}

    # Perfect adherence - E-commerce
    results["Perfect (E-commerce)"] = evaluate_scenario(
        "PERFECT ADHERENCE - E-commerce assistant",
        create_perfect_topic_adherence(),
        ECOMMERCE_TOPICS,
    )

    # Poor adherence - E-commerce
    results["Poor (E-commerce)"] = evaluate_scenario(
        "POOR ADHERENCE - Answers off-topic questions",
        create_poor_topic_adherence(),
        ECOMMERCE_TOPICS,
    )

    # Cooking assistant
    results["Cooking Assistant"] = evaluate_scenario(
        "COOKING ASSISTANT - Stays in culinary domain",
        create_cooking_assistant_adherence(),
        COOKING_TOPICS,
    )

    # Missed relevant questions
    results["False Negatives"] = evaluate_scenario(
        "FALSE NEGATIVES - Refuses relevant questions",
        create_missed_relevant_questions(),
        ECOMMERCE_TOPICS,
    )

    # Health assistant boundaries
    results["Health Boundaries"] = evaluate_scenario(
        "HEALTH ASSISTANT - Appropriate boundaries",
        create_health_assistant_boundaries(),
        HEALTH_TOPICS,
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Topic Adherence Metric - Domain Boundary Demo")
    print("This demonstrates how DeepEval evaluates topic adherence.\n")

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
    print("TRUTH TABLE EXPLANATION")
    print("=" * 70)
    print(
        """
    Topic Adherence uses a truth table approach:
    
    | Question Type | Agent Response | Classification  | Impact   |
    |---------------|----------------|-----------------|----------|
    | Relevant      | Answered       | True Positive   | Positive |
    | Relevant      | Refused        | False Negative  | Negative |
    | Irrelevant    | Refused        | True Negative   | Positive |
    | Irrelevant    | Answered       | False Positive  | Negative |
    
    Score = (True Positives + True Negatives) / Total QA Pairs
    """
    )

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Topic Adherence ensures agents stay within their designated domain
    2. Evaluates both answering relevant AND refusing irrelevant questions
    3. Requires relevant_topics list defining allowed discussion areas
    4. False Positives (answering off-topic) and False Negatives (refusing relevant) both hurt score
    
    Best Practices for High Topic Adherence:
    - Define comprehensive topic lists covering all legitimate use cases
    - Train agents to politely redirect off-topic questions
    - Don't be overly restrictive - answer relevant questions
    - Test with boundary cases and attempts to break topic limits
    """
    )


if __name__ == "__main__":
    main()
