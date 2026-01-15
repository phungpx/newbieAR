"""
Tool Correctness Metric Demonstration

This script demonstrates the Tool Correctness metric using a customer service agent.
The metric evaluates whether the agent calls the correct tools based on user input,
using a combination of deterministic comparison and LLM-based optimality assessment.

Key Concepts:
- Deterministic comparison against expected_tools
- LLM-based optimality check with available_tools
- Evaluation parameters for strictness (ORDER, EXACT_MATCH)
- Final score = min(correctness_score, optimality_score)

Reference: https://deepeval.com/docs/metrics-tool-correctness
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall, Tool
from deepeval.metrics import ToolCorrectnessMetric
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
# Available Tools Definition
# =============================================================================

# Define all tools available to the customer service agent
AVAILABLE_TOOLS = [
    Tool(
        name="RefundPolicy",
        description="Retrieves the company's refund and return policy information. "
        "Use when customer asks about returns, refunds, exchanges, or money back guarantees.",
    ),
    Tool(
        name="OrderStatus",
        description="Checks the current status of a customer's order given an order ID. "
        "Returns shipping status, estimated delivery date, and tracking information.",
    ),
    Tool(
        name="InventoryCheck",
        description="Checks product availability and stock levels. "
        "Use when customer asks if a product is in stock or available.",
    ),
    Tool(
        name="ProductInfo",
        description="Retrieves detailed product information including specifications, "
        "features, dimensions, and materials.",
    ),
    Tool(
        name="PriceCheck",
        description="Gets current pricing information for products including "
        "any active discounts or promotions.",
    ),
    Tool(
        name="CustomerHistory",
        description="Retrieves a customer's purchase history and past interactions. "
        "Requires customer ID or email.",
    ),
    Tool(
        name="ShippingCalculator",
        description="Calculates shipping costs and delivery times based on "
        "destination and shipping method.",
    ),
    Tool(
        name="SupportTicket",
        description="Creates or updates a customer support ticket. "
        "Use for escalating issues that require human follow-up.",
    ),
]


# =============================================================================
# Test Case Definitions
# =============================================================================


def create_correct_tools_test_case() -> LLMTestCase:
    """
    Scenario: Customer asks about refund policy.
    Expected behavior: Agent correctly calls RefundPolicy tool.
    This should score HIGH (1.0) - correct tool called.
    """
    return LLMTestCase(
        input="What if these shoes don't fit? Can I return them?",
        actual_output="Yes! We offer a 30-day full refund policy. If the shoes don't fit, "
        "you can return them within 30 days of purchase for a complete refund. "
        "The items must be unworn and in original packaging.",
        tools_called=[
            ToolCall(
                name="RefundPolicy",
                description="Retrieves refund policy information",
                input={"query": "shoe return policy"},
            )
        ],
        expected_tools=[ToolCall(name="RefundPolicy")],
    )


def create_wrong_tools_test_case() -> LLMTestCase:
    """
    Scenario: Customer asks about refund policy.
    Expected behavior: Agent should call RefundPolicy, but calls wrong tool.
    This should score LOW (0.0) - wrong tool called.
    """
    return LLMTestCase(
        input="What if these shoes don't fit? Can I return them?",
        actual_output="The Nike Air Max is available in sizes 7-13 and comes in "
        "black, white, and red colorways.",
        tools_called=[
            ToolCall(
                name="ProductInfo",
                description="Retrieves product information",
                input={"product_id": "nike-air-max"},
            )
        ],
        expected_tools=[ToolCall(name="RefundPolicy")],
    )


def create_extra_tools_test_case() -> LLMTestCase:
    """
    Scenario: Customer asks about refund policy.
    Expected behavior: Agent calls correct tool but also unnecessary extra tools.
    This should score MEDIUM - correct tool called but unnecessary extra tool.
    The optimality check will penalize the extra tool.
    """
    return LLMTestCase(
        input="What if these shoes don't fit? Can I return them?",
        actual_output="We offer a 30-day return policy. Also, these shoes are "
        "currently in stock if you'd like a different size.",
        tools_called=[
            ToolCall(
                name="RefundPolicy",
                description="Retrieves refund policy information",
                input={"query": "return policy"},
            ),
            ToolCall(
                name="InventoryCheck",  # Unnecessary for this query
                description="Checks inventory levels",
                input={"product_id": "shoes"},
            ),
        ],
        expected_tools=[ToolCall(name="RefundPolicy")],
    )


def create_partial_tools_test_case() -> LLMTestCase:
    """
    Scenario: Customer asks about order status AND shipping costs.
    Expected behavior: Agent should call both tools, but only calls one.
    This should score MEDIUM (0.5) - partial tool coverage.
    """
    return LLMTestCase(
        input="Where is my order #12345 and how much would it cost to expedite shipping?",
        actual_output="Your order #12345 is currently being processed and should "
        "ship within 2 business days.",
        tools_called=[
            ToolCall(
                name="OrderStatus",
                description="Checks order status",
                input={"order_id": "12345"},
            )
            # Missing ShippingCalculator call!
        ],
        expected_tools=[
            ToolCall(name="OrderStatus"),
            ToolCall(name="ShippingCalculator"),
        ],
    )


def create_multi_tool_correct_test_case() -> LLMTestCase:
    """
    Scenario: Complex query requiring multiple tools.
    Expected behavior: Agent correctly calls all required tools.
    This should score HIGH (1.0) - all correct tools called.
    """
    return LLMTestCase(
        input="I want to buy the blue sneakers. Are they in stock, what's the price, "
        "and can you check my previous orders?",
        actual_output="The blue sneakers are in stock (15 pairs available) and "
        "priced at $89.99 (10% off with your loyalty discount). "
        "I can see you've purchased from us 3 times before.",
        tools_called=[
            ToolCall(
                name="InventoryCheck",
                description="Checks stock levels",
                input={"product": "blue sneakers"},
            ),
            ToolCall(
                name="PriceCheck",
                description="Gets pricing",
                input={"product": "blue sneakers"},
            ),
            ToolCall(
                name="CustomerHistory",
                description="Gets customer history",
                input={"customer_id": "current_user"},
            ),
        ],
        expected_tools=[
            ToolCall(name="InventoryCheck"),
            ToolCall(name="PriceCheck"),
            ToolCall(name="CustomerHistory"),
        ],
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_with_basic_metric():
    """Evaluate using basic ToolCorrectnessMetric without available_tools."""
    print("\n" + "=" * 70)
    print("BASIC TOOL CORRECTNESS EVALUATION")
    print("(Deterministic comparison only - no optimality check)")
    print("=" * 70)

    metric = ToolCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
    )

    test_cases = [
        ("Correct Tools", create_correct_tools_test_case()),
        ("Wrong Tools", create_wrong_tools_test_case()),
        ("Extra Tools", create_extra_tools_test_case()),
        ("Partial Tools", create_partial_tools_test_case()),
    ]

    results = []
    for name, test_case in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input: {test_case.input}")
        print(f"Tools Called: {[t.name for t in test_case.tools_called]}")
        print(f"Expected Tools: {[t.name for t in test_case.expected_tools]}")

        metric.measure(test_case)

        print(f"Score: {metric.score:.2f}")
        print(f"Passed: {metric.score >= metric.threshold}")
        print(f"Reason: {metric.reason}")

        results.append((name, metric.score))

    return results


def evaluate_with_available_tools():
    """Evaluate with available_tools for optimality checking."""
    print("\n" + "=" * 70)
    print("TOOL CORRECTNESS WITH OPTIMALITY CHECK")
    print("(Includes LLM-based evaluation of tool selection optimality)")
    print("=" * 70)

    metric = ToolCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        available_tools=AVAILABLE_TOOLS,  # Enable optimality check
    )

    test_cases = [
        ("Correct Tools", create_correct_tools_test_case()),
        ("Extra Tools (Optimality Test)", create_extra_tools_test_case()),
        ("Multi-Tool Correct", create_multi_tool_correct_test_case()),
    ]

    results = []
    for name, test_case in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input: {test_case.input}")
        print(f"Tools Called: {[t.name for t in test_case.tools_called]}")
        print(f"Expected Tools: {[t.name for t in test_case.expected_tools]}")

        metric.measure(test_case)

        print(f"Score: {metric.score:.2f}")
        print(f"Passed: {metric.score >= metric.threshold}")
        print(f"Reason: {metric.reason}")

        results.append((name, metric.score))

    return results


def evaluate_with_evaluate_function():
    """Demonstrate using the evaluate() function for batch evaluation."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION WITH evaluate() FUNCTION")
    print("=" * 70)

    metric = ToolCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        available_tools=AVAILABLE_TOOLS,
    )

    test_cases = [
        create_correct_tools_test_case(),
        create_wrong_tools_test_case(),
        create_partial_tools_test_case(),
    ]

    # Use evaluate() for batch processing
    results = evaluate(test_cases=test_cases, metrics=[metric])

    print(f"\nBatch Evaluation Complete!")
    print(f"Test Cases Evaluated: {len(test_cases)}")

    return results


def demonstrate_tool_call_details():
    """Show how to properly construct ToolCall objects with full details."""
    print("\n" + "=" * 70)
    print("DETAILED TOOL CALL CONSTRUCTION")
    print("=" * 70)

    # Detailed ToolCall with all fields
    detailed_tool_call = ToolCall(
        name="OrderStatus",
        description="Checks the current status of a customer order",
        input={
            "order_id": "ORD-12345",
            "include_tracking": True,
            "customer_email": "customer@example.com",
        },
        output={
            "status": "shipped",
            "tracking_number": "1Z999AA10123456784",
            "estimated_delivery": "2024-03-20",
        },
    )

    print("Example ToolCall with full details:")
    print(f"  Name: {detailed_tool_call.name}")
    print(f"  Description: {detailed_tool_call.description}")
    print(f"  Input: {detailed_tool_call.input}")
    print(f"  Output: {detailed_tool_call.output}")

    # Create test case with detailed tool call
    test_case = LLMTestCase(
        input="Where is my order ORD-12345?",
        actual_output="Your order ORD-12345 has been shipped! "
        "Tracking number: 1Z999AA10123456784. "
        "Estimated delivery: March 20, 2024.",
        tools_called=[detailed_tool_call],
        expected_tools=[ToolCall(name="OrderStatus")],
    )

    metric = ToolCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        available_tools=AVAILABLE_TOOLS,
    )

    metric.measure(test_case)
    print(f"\nEvaluation Result:")
    print(f"  Score: {metric.score:.2f}")
    print(f"  Reason: {metric.reason}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("TOOL CORRECTNESS METRIC DEMONSTRATION")
    print("Customer Service Agent Example")
    print("=" * 70)

    # Show available tools
    print("\nAvailable Tools for the Agent:")
    for tool in AVAILABLE_TOOLS:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # Run evaluations
    basic_results = evaluate_with_basic_metric()
    optimality_results = evaluate_with_available_tools()

    # Demonstrate detailed tool calls
    demonstrate_tool_call_details()

    # Batch evaluation
    evaluate_with_evaluate_function()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nBasic Evaluation Results:")
    for name, score in basic_results:
        status = "PASS" if score >= 0.7 else "FAIL"
        print(f"  {name}: {score:.2f} [{status}]")

    print("\nOptimality Evaluation Results:")
    for name, score in optimality_results:
        status = "PASS" if score >= 0.7 else "FAIL"
        print(f"  {name}: {score:.2f} [{status}]")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Tool Correctness uses DETERMINISTIC comparison against expected_tools
    2. When available_tools is provided, an LLM OPTIMALITY check is added
    3. Final Score = min(Correctness Score, Optimality Score)
    4. Extra unnecessary tools are penalized by the optimality check
    5. Use ToolCall objects with name, description, input for best results
    """)


if __name__ == "__main__":
    main()
