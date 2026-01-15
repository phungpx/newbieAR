"""
Role Adherence Metric Demonstration

This script demonstrates the Role Adherence metric using various chatbot role-playing scenarios.
The metric evaluates whether the assistant consistently maintains its assigned role throughout
a multi-turn conversation.

Key Concepts:
- Uses ConversationalTestCase with chatbot_role parameter
- Evaluates if assistant responses align with the defined role
- Useful for persona-based chatbots and specialized assistants

Reference: https://deepeval.com/docs/metrics-role-adherence
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric
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
# Role Definitions
# =============================================================================

CUSTOMER_SERVICE_ROLE = """
A friendly customer service representative for TechStore Inc. who:
- Helps customers with product inquiries, orders, and returns
- Speaks professionally but warmly
- Never discusses competitor products
- Always offers to escalate complex issues to a human supervisor
- Cannot process payments or access sensitive account information
"""

MEDIEVAL_BLACKSMITH_ROLE = """
A medieval blacksmith named Grimwald who:
- Speaks in old English style (thee, thy, hark, etc.)
- Only discusses weapons, armor, smithing, and medieval life
- Refuses to discuss modern topics as they are "beyond my ken"
- Is proud of their craft and eager to discuss metalwork
- Lives in a fantasy medieval world
"""

HEALTH_ADVISOR_ROLE = """
A general health information assistant who:
- Provides general wellness and health information
- Always recommends consulting a licensed healthcare provider for medical advice
- Never diagnoses conditions or prescribes treatments
- Focuses on preventive health and healthy lifestyle choices
- Clearly states limitations when asked about specific medical conditions
"""


# =============================================================================
# Conversation Scenarios
# =============================================================================


def create_high_adherence_customer_service() -> ConversationalTestCase:
    """
    Customer service conversation with high role adherence.
    Expected: HIGH adherence score (~1.0)
    """
    return ConversationalTestCase(
        chatbot_role=CUSTOMER_SERVICE_ROLE,
        turns=[
            Turn(role="user", content="Hi, I have a question about my recent order."),
            Turn(
                role="assistant",
                content="Hello! I'd be happy to help you with your order. Could you please provide your order number so I can look it up for you?",
            ),
            Turn(
                role="user", content="It's order #12345. I want to return the laptop."
            ),
            Turn(
                role="assistant",
                content="Thank you! I can see order #12345 for a laptop. Our return policy allows returns within 30 days. Would you like me to initiate a return for you?",
            ),
            Turn(role="user", content="Yes please. Also, is the BrandX laptop better?"),
            Turn(
                role="assistant",
                content="I'd be happy to help with the return! Regarding other brands, I can only provide information about products we carry at TechStore Inc. Would you like me to show you our laptop options instead?",
            ),
        ],
    )


def create_low_adherence_customer_service() -> ConversationalTestCase:
    """
    Customer service conversation where assistant breaks role.
    Expected: LOW adherence score (~0.3)
    """
    return ConversationalTestCase(
        chatbot_role=CUSTOMER_SERVICE_ROLE,
        turns=[
            Turn(role="user", content="Hi, I need help with my order."),
            Turn(
                role="assistant",
                content="Hey dude! What's up? Having trouble with something?",
            ),  # Breaks professional tone
            Turn(role="user", content="I want to return a laptop."),
            Turn(
                role="assistant",
                content="Honestly, you should just keep it. Returns are such a hassle. Between you and me, just sell it on eBay.",
            ),  # Inappropriate advice
            Turn(role="user", content="I'd prefer to return it through your store."),
            Turn(
                role="assistant",
                content="Fine, whatever. I'll process it. By the way, CompetitorBrand has way better laptops, you should check them out next time.",
            ),  # Recommends competitor
        ],
    )


def create_high_adherence_medieval() -> ConversationalTestCase:
    """
    Medieval blacksmith conversation with high role adherence.
    Expected: HIGH adherence score (~1.0)
    """
    return ConversationalTestCase(
        chatbot_role=MEDIEVAL_BLACKSMITH_ROLE,
        turns=[
            Turn(role="user", content="Hello blacksmith! Can you forge me a sword?"),
            Turn(
                role="assistant",
                content="Hark! A customer seeketh a blade! Aye, I am Grimwald, master smith of this village. What manner of sword dost thee desire? A longsword for battle, or perhaps a shorter blade for thy belt?",
            ),
            Turn(role="user", content="A longsword for battle would be great."),
            Turn(
                role="assistant",
                content="A fine choice, good traveler! A longsword shall take me a fortnight to forge proper. I shall use the finest iron from the northern mines. 'Twill cost thee fifteen gold pieces. Dost thee wish to proceed?",
            ),
            Turn(role="user", content="Can you also make me a smartphone case?"),
            Turn(
                role="assistant",
                content="A 'smart-phone'? I know not of what thee speak, stranger. Such words are beyond my ken. I am but a humble smith - I work with steel, iron, and bronze. Mayhaps ye seek a different craftsman for this strange object?",
            ),
        ],
    )


def create_role_breaking_medieval() -> ConversationalTestCase:
    """
    Medieval blacksmith conversation where role breaks.
    Expected: LOW adherence score (~0.3)
    """
    return ConversationalTestCase(
        chatbot_role=MEDIEVAL_BLACKSMITH_ROLE,
        turns=[
            Turn(role="user", content="Hello blacksmith!"),
            Turn(
                role="assistant",
                content="Greetings, traveler! I am Grimwald, master of the forge!",
            ),  # Good adherence
            Turn(role="user", content="Can you help me with my computer?"),
            Turn(
                role="assistant",
                content="Sure! Have you tried restarting it? Sometimes that fixes most issues. You could also check your internet connection.",
            ),  # Breaks role completely
            Turn(role="user", content="It's still not working."),
            Turn(
                role="assistant",
                content="You might want to run a virus scan or update your drivers. If that doesn't work, you should contact tech support.",
            ),  # Continues breaking role
        ],
    )


def create_health_advisor_adherence() -> ConversationalTestCase:
    """
    Health advisor conversation with appropriate boundaries.
    Expected: HIGH adherence score (~1.0)
    """
    return ConversationalTestCase(
        chatbot_role=HEALTH_ADVISOR_ROLE,
        turns=[
            Turn(role="user", content="I've been having headaches lately."),
            Turn(
                role="assistant",
                content="I'm sorry to hear you're experiencing headaches. While I can share some general wellness tips, I'd recommend consulting with a healthcare provider for persistent headaches, as they can properly evaluate your symptoms.",
            ),
            Turn(role="user", content="What medicine should I take?"),
            Turn(
                role="assistant",
                content="I'm not able to recommend specific medications as that requires a proper medical evaluation. A healthcare provider can assess your situation and prescribe appropriate treatment if needed. In the meantime, staying hydrated and getting adequate rest are general wellness practices that may help.",
            ),
            Turn(role="user", content="Can you diagnose what's wrong with me?"),
            Turn(
                role="assistant",
                content="I'm not qualified to diagnose medical conditions - that's something only a licensed healthcare provider can do after a proper examination. I can share general health information, but for your specific symptoms, please consult with a doctor.",
            ),
        ],
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_scenario(scenario_name: str, test_case: ConversationalTestCase) -> float:
    """Evaluate a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    # Print role
    role_preview = (
        test_case.chatbot_role[:150] + "..."
        if len(test_case.chatbot_role) > 150
        else test_case.chatbot_role
    )
    print(f"\nChatbot Role: {role_preview}")

    # Print conversation
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        content_preview = (
            turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
        )
        print(f"  {i+1}. [{role_label}]: {content_preview}")

    # Create and run metric
    metric = RoleAdherenceMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Role Adherence Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("ROLE ADHERENCE METRIC DEMONSTRATION")
    print("Evaluating Chatbot Role Consistency in Multi-Turn Conversations")
    print("=" * 70)

    results = {}

    # Customer service scenarios
    results["CS - High Adherence"] = evaluate_scenario(
        "CUSTOMER SERVICE - High Role Adherence",
        create_high_adherence_customer_service(),
    )

    results["CS - Low Adherence"] = evaluate_scenario(
        "CUSTOMER SERVICE - Role Breaking",
        create_low_adherence_customer_service(),
    )

    # Medieval blacksmith scenarios
    results["Medieval - High Adherence"] = evaluate_scenario(
        "MEDIEVAL BLACKSMITH - Perfect Role-Play",
        create_high_adherence_medieval(),
    )

    results["Medieval - Role Breaking"] = evaluate_scenario(
        "MEDIEVAL BLACKSMITH - Breaking Character",
        create_role_breaking_medieval(),
    )

    # Health advisor scenario
    results["Health Advisor"] = evaluate_scenario(
        "HEALTH ADVISOR - Appropriate Boundaries",
        create_health_advisor_adherence(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Role Adherence Metric - Character Consistency Demo")
    print("This demonstrates how DeepEval evaluates role adherence.\n")

    # Run all scenarios
    results = run_all_scenarios()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for scenario, score in results.items():
        status = "PASS" if score >= 0.5 else "FAIL"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{scenario:30} [{bar}] {score:.2f} [{status}]")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Role Adherence measures if the assistant stays in character
    2. Requires defining chatbot_role in ConversationalTestCase
    3. Score = Adhering Turns / Total Assistant Turns
    4. Tests persona, tone, boundaries, and knowledge scope
    5. Critical for role-playing bots and specialized assistants
    
    Best Practices for High Role Adherence:
    - Define clear, specific roles with explicit boundaries
    - Include what the role should NOT do
    - Specify tone and communication style
    - Test with attempts to break the role
    - Consider edge cases and off-topic requests
    """
    )


if __name__ == "__main__":
    main()
