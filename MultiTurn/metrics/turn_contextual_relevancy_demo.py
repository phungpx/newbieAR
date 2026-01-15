"""
Turn Contextual Relevancy Metric Demonstration

This script demonstrates the Turn Contextual Relevancy metric for RAG chatbots.
The metric evaluates whether the retrieval context contains relevant information
to address the user's input throughout a conversation.

Key Concepts:
- Uses ConversationalTestCase with retrieval_context in Turn objects
- Measures how much of retrieved content is actually relevant
- Helps identify noise in retrieval results

Reference: https://deepeval.com/docs/metrics-turn-contextual-relevancy
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric
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


def create_highly_relevant_context() -> ConversationalTestCase:
    """
    Conversation where all retrieved context is relevant to user query.
    Expected: HIGH contextual relevancy score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="How do I reset my password?"),
            Turn(
                role="assistant",
                content="To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link sent to you.",
                retrieval_context=[
                    "Password Reset: Click 'Forgot Password' on the login page to initiate a password reset.",
                    "Reset Link: After requesting a reset, you'll receive an email with a secure reset link within 5 minutes.",
                    "Password Requirements: New passwords must be at least 8 characters with one uppercase letter and one number.",
                    "Reset Link Expiry: Password reset links expire after 24 hours for security purposes.",
                ],
            ),
        ]
    )


def create_noisy_context() -> ConversationalTestCase:
    """
    Conversation where retrieval context contains mostly irrelevant information.
    Expected: LOW contextual relevancy score (~0.2-0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="How do I reset my password?"),
            Turn(
                role="assistant",
                content="To reset your password, use the Forgot Password link on the login page.",
                retrieval_context=[
                    "Company Overview: We are a leading technology company founded in 2010.",
                    "Career Opportunities: Join our team! We're hiring engineers and designers.",
                    "Office Locations: Our headquarters is in San Francisco with offices in NYC and London.",
                    "Password Reset: Use the 'Forgot Password' link on the login page to reset.",
                    "Holiday Schedule: Our offices are closed on federal holidays.",
                    "Parking Information: Free parking available in the main lot.",
                ],
            ),
        ]
    )


def create_mixed_relevancy_context() -> ConversationalTestCase:
    """
    Conversation with mix of relevant and irrelevant context.
    Expected: MEDIUM contextual relevancy score (~0.5-0.6)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What are your shipping options?"),
            Turn(
                role="assistant",
                content="We offer standard shipping (5-7 days, free), express (2-3 days, $9.99), and overnight ($19.99).",
                retrieval_context=[
                    "Shipping Methods: Standard (5-7 days), Express (2-3 days), Overnight (1 day).",
                    "Shipping Costs: Standard is free, Express is $9.99, Overnight is $19.99.",
                    "Return Policy: Items can be returned within 30 days.",  # Not relevant
                    "Gift Cards: We sell gift cards in denominations from $25 to $500.",  # Not relevant
                    "Tracking: All shipments include tracking numbers.",
                    "Store Hours: Open Monday-Saturday 9AM-9PM.",  # Not relevant
                ],
            ),
        ]
    )


def create_multi_turn_relevancy() -> ConversationalTestCase:
    """
    Multi-turn conversation with varying relevancy across turns.
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What laptops do you have?"),
            Turn(
                role="assistant",
                content="We carry Dell, HP, Lenovo, and ASUS laptops ranging from $499 to $2,499.",
                retrieval_context=[
                    # Good relevancy
                    "Laptop Brands: We carry Dell, HP, Lenovo, and ASUS laptops.",
                    "Price Range: Laptops start at $499 and go up to $2,499.",
                    "Laptop Specs: Available with Intel or AMD processors, 8-32GB RAM.",
                ],
            ),
            Turn(role="user", content="What's the warranty on the Dell XPS?"),
            Turn(
                role="assistant",
                content="The Dell XPS comes with a 1-year warranty, extendable to 3 years.",
                retrieval_context=[
                    # Mixed relevancy
                    "Dell XPS Warranty: 1-year manufacturer warranty, can be extended to 3 years.",
                    "General Store Policy: All sales are final on clearance items.",  # Not relevant
                    "Laptop Accessories: We sell cases, mice, and chargers.",  # Not relevant
                    "Dell Support: Dell offers 24/7 phone support for XPS owners.",
                ],
            ),
        ]
    )


def create_completely_irrelevant() -> ConversationalTestCase:
    """
    Scenario where retrieval completely failed to get relevant documents.
    Expected: VERY LOW contextual relevancy score (~0.0-0.1)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is the refund policy for online orders?"),
            Turn(
                role="assistant",
                content="I don't have specific information about refunds in my current context.",
                retrieval_context=[
                    "Company History: Founded in 1985, we've grown to 500 stores nationwide.",
                    "Executive Team: Our CEO has 20 years of retail experience.",
                    "Sustainability: We're committed to reducing our carbon footprint by 50% by 2030.",
                    "Community Involvement: We donate 5% of profits to local charities.",
                ],
            ),
        ]
    )


def create_optimal_retrieval() -> ConversationalTestCase:
    """
    Scenario with perfectly focused, relevant retrieval results.
    Expected: HIGH contextual relevancy score (~0.95-1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="How do I track my order?"),
            Turn(
                role="assistant",
                content="You can track your order using the tracking number in your confirmation email or through your account dashboard.",
                retrieval_context=[
                    "Order Tracking: Track your order using the tracking number from your confirmation email.",
                    "Account Dashboard: Log into your account to view real-time order status and tracking.",
                    "Tracking Updates: You'll receive email updates when your order ships and is out for delivery.",
                    "Delivery Notifications: Enable push notifications in the app for delivery alerts.",
                ],
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

    # Print conversation with context
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        print(f"  {i+1}. [{role_label}]: {turn.content[:70]}...")
        if turn.retrieval_context:
            print(f"       Retrieved {len(turn.retrieval_context)} documents:")
            for j, ctx in enumerate(turn.retrieval_context):
                print(f"         {j+1}. {ctx[:55]}...")

    # Create and run metric
    metric = TurnContextualRelevancyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Contextual Relevancy Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("TURN CONTEXTUAL RELEVANCY METRIC DEMONSTRATION")
    print("Evaluating Retrieval Relevance in RAG Conversations")
    print("=" * 70)

    results = {}

    # Highly relevant
    results["Highly Relevant"] = evaluate_scenario(
        "HIGHLY RELEVANT - All context addresses query",
        create_highly_relevant_context(),
    )

    # Noisy context
    results["Noisy Context"] = evaluate_scenario(
        "NOISY - Mostly irrelevant documents retrieved",
        create_noisy_context(),
    )

    # Mixed relevancy
    results["Mixed"] = evaluate_scenario(
        "MIXED - Some relevant, some irrelevant",
        create_mixed_relevancy_context(),
    )

    # Multi-turn
    results["Multi-Turn"] = evaluate_scenario(
        "MULTI-TURN - Relevancy varies per turn",
        create_multi_turn_relevancy(),
    )

    # Completely irrelevant
    results["Irrelevant"] = evaluate_scenario(
        "IRRELEVANT - Retrieval completely missed",
        create_completely_irrelevant(),
    )

    # Optimal retrieval
    results["Optimal"] = evaluate_scenario(
        "OPTIMAL - Perfect retrieval results",
        create_optimal_retrieval(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Turn Contextual Relevancy Metric - Retrieval Quality Demo")
    print("This demonstrates how DeepEval evaluates retrieval relevance.\n")

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
    print("RELEVANCY FORMULA")
    print("=" * 70)
    print(
        """
    Contextual Relevancy = Relevant Statements / Total Statements
    
    Process:
    1. Extract individual statements from each context document
    2. Evaluate if each statement is relevant to the user's input
    3. Calculate ratio of relevant statements
    
    Example for query "How do I reset my password?":
    
    Context Statements:
    1. "Click Forgot Password to reset"  -> Relevant? YES
    2. "Reset links expire in 24 hours"  -> Relevant? YES
    3. "We were founded in 2010"         -> Relevant? NO
    4. "Free parking available"          -> Relevant? NO
    
    Relevancy = 2/4 = 0.50
    """
    )

    print("\n" + "=" * 70)
    print("METRIC COMPARISON")
    print("=" * 70)
    print(
        """
    | Metric                | Focus                              | Question Answered                    |
    |-----------------------|------------------------------------|--------------------------------------|
    | Contextual Relevancy  | Is context relevant to query?      | "Is this context useful?"            |
    | Contextual Precision  | Is relevant context ranked higher? | "Are good docs at the top?"          |
    | Contextual Recall     | Does context cover expected info?  | "Is all needed info present?"        |
    | Faithfulness          | Is response faithful to context?   | "Does response match context?"       |
    """
    )

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Contextual Relevancy measures how much retrieved content is useful
    2. High relevancy = efficient use of context window
    3. Low relevancy = noise that can confuse the model
    4. Helps optimize retrieval filters and thresholds
    5. Critical for cost optimization (fewer irrelevant tokens processed)
    
    Best Practices for High Relevancy:
    - Use relevance filtering after retrieval
    - Set appropriate similarity thresholds
    - Implement smaller, focused document chunks
    - Consider domain-specific embedding models
    - Remove or downrank consistently irrelevant content
    """
    )


if __name__ == "__main__":
    main()
