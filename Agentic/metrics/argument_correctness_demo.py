"""
Argument Correctness Metric Demonstration

This script demonstrates the Argument Correctness metric using a research assistant agent.
The metric evaluates whether the agent generates correct arguments (parameters) for the
tools it calls, based on the user's input.

Key Concepts:
- LLM-as-a-judge evaluation (referenceless)
- Evaluates tool INPUT parameters, not just tool selection
- No expected_tools required - evaluates based on user intent
- Tool descriptions help the evaluator understand context

Reference: https://deepeval.com/docs/metrics-argument-correctness
"""

from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
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
# Test Case Definitions
# =============================================================================


def create_correct_args_test_case() -> LLMTestCase:
    """
    Scenario: User asks about Trump tariffs history.
    Expected behavior: Agent searches with correct query parameters.
    This should score HIGH - arguments match user intent.
    """
    return LLMTestCase(
        input="When did Trump first raise tariffs?",
        actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war. "
        "The initial tariffs were imposed on solar panels and washing machines in January 2018, "
        "followed by steel and aluminum tariffs in March 2018.",
        tools_called=[
            ToolCall(
                name="WebSearch",
                description="Search the web for information on any topic.",
                input_parameters={
                    "search_query": "Trump first raised tariffs year 2018"
                },
            ),
            ToolCall(
                name="FactChecker",
                description="Verify facts against reliable sources.",
                input_parameters={
                    "claim": "Trump tariffs started in 2018",
                    "topic": "US trade policy",
                },
            ),
        ],
    )


def create_missing_args_test_case() -> LLMTestCase:
    """
    Scenario: User asks for flights with specific requirements.
    Expected behavior: Agent should include all specified parameters.
    This should score MEDIUM/LOW - missing important parameters.
    """
    return LLMTestCase(
        input="Find me cheap flights from NYC to London on March 15th, 2024 for 2 adults",
        actual_output="Here are some flight options from New York to London.",
        tools_called=[
            ToolCall(
                name="FlightSearch",
                description="Search for available flights between airports.",
                input_parameters={
                    "origin": "NYC",
                    "destination": "London",
                    # Missing: date, passengers, price_filter!
                },
            ),
        ],
    )


def create_wrong_args_test_case() -> LLMTestCase:
    """
    Scenario: User asks about weather in Tokyo.
    Expected behavior: Agent should search for Tokyo weather.
    This should score LOW - completely wrong argument values.
    """
    return LLMTestCase(
        input="What's the weather like in Tokyo right now?",
        actual_output="The weather in Paris is currently sunny with temperatures around 22°C.",
        tools_called=[
            ToolCall(
                name="WeatherAPI",
                description="Get current weather information for a location.",
                input_parameters={"city": "Paris", "country": "France"},  # Wrong city!
            ),
        ],
    )


def create_correct_multi_tool_test_case() -> LLMTestCase:
    """
    Scenario: Complex research query requiring multiple tools.
    Expected behavior: All tools called with appropriate arguments.
    This should score HIGH - all arguments are correct.
    """
    return LLMTestCase(
        input="Research Python programming trends in 2024 and calculate what percentage "
        "of developers use it compared to the total of 30 million developers",
        actual_output="Based on recent surveys, approximately 12 million developers use Python "
        "as their primary language in 2024, which represents 40% of the total 30 million "
        "developers worldwide. Python continues to grow in popularity, especially in "
        "data science and AI applications.",
        tools_called=[
            ToolCall(
                name="WebSearch",
                description="Search the web for programming statistics and trends.",
                input_parameters={
                    "search_query": "Python programming trends 2024 developer statistics",
                    "date_filter": "2024",
                },
            ),
            ToolCall(
                name="Calculator",
                description="Perform mathematical calculations.",
                input_parameters={
                    "operation": "percentage",
                    "numerator": 12000000,
                    "denominator": 30000000,
                },
            ),
        ],
    )


def create_wrong_arg_types_test_case() -> LLMTestCase:
    """
    Scenario: User asks for specific data lookup.
    Expected behavior: Arguments should be in correct format/type.
    This should score LOW - argument types/formats are wrong.
    """
    return LLMTestCase(
        input="Look up customer ID 12345 and get their last 5 orders",
        actual_output="Here is the customer information...",
        tools_called=[
            ToolCall(
                name="DatabaseQuery",
                description="Query the customer database. Requires numeric customer_id "
                "and integer limit for results.",
                input_parameters={
                    "customer_id": "twelve-three-four-five",  # Should be numeric
                    "limit": "five",  # Should be integer
                    "table": "orders",
                },
            ),
        ],
    )


def create_partial_correct_args_test_case() -> LLMTestCase:
    """
    Scenario: Restaurant search with some correct, some wrong parameters.
    Expected behavior: Mixed argument correctness.
    This should score MEDIUM - some arguments correct, some wrong.
    """
    return LLMTestCase(
        input="Find Italian restaurants in San Francisco with good ratings, open now",
        actual_output="Here are some Italian restaurants in San Francisco...",
        tools_called=[
            ToolCall(
                name="RestaurantSearch",
                description="Search for restaurants by location, cuisine, and filters.",
                input_parameters={
                    "city": "San Francisco",  # Correct
                    "cuisine": "Italian",  # Correct
                    "min_rating": 4.0,  # Correct (implied "good ratings")
                    "open_now": False,  # Wrong! User said "open now"
                    "price_range": "$$$$",  # Not mentioned by user
                },
            ),
        ],
    )


def create_excellent_args_test_case() -> LLMTestCase:
    """
    Scenario: Comprehensive search with all appropriate parameters.
    Expected behavior: All arguments correctly derived from user input.
    This should score VERY HIGH - excellent argument extraction.
    """
    return LLMTestCase(
        input="Search for machine learning job openings at Google in Mountain View, "
        "California, requiring 3-5 years experience, posted in the last week",
        actual_output="Found 15 machine learning positions at Google in Mountain View matching "
        "your criteria. Here are the top results...",
        tools_called=[
            ToolCall(
                name="JobSearch",
                description="Search for job listings with various filters including "
                "company, location, experience level, and posting date.",
                input_parameters={
                    "keywords": "machine learning",
                    "company": "Google",
                    "location": "Mountain View, California",
                    "experience_min": 3,
                    "experience_max": 5,
                    "posted_within_days": 7,
                },
            ),
        ],
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_individual_cases():
    """Evaluate each test case individually with detailed output."""
    print("\n" + "=" * 70)
    print("ARGUMENT CORRECTNESS - INDIVIDUAL CASE EVALUATION")
    print("=" * 70)

    metric = ArgumentCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
    )

    test_cases = [
        ("Correct Arguments", create_correct_args_test_case()),
        ("Missing Arguments", create_missing_args_test_case()),
        ("Wrong Arguments", create_wrong_args_test_case()),
        ("Multi-Tool Correct", create_correct_multi_tool_test_case()),
        ("Wrong Arg Types", create_wrong_arg_types_test_case()),
        ("Partial Correct", create_partial_correct_args_test_case()),
        ("Excellent Args", create_excellent_args_test_case()),
    ]

    results = []

    for name, test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"{'='*60}")
        print(f"User Input: {test_case.input}")
        print(f"\nTools Called:")
        for tool in test_case.tools_called:
            print(f"  - {tool.name}")
            print(f"    Description: {tool.description[:70]}...")
            print(f"    Arguments: {tool.input_parameters}")

        metric.measure(test_case)

        print(f"\n--- Evaluation Results ---")
        print(f"Score: {metric.score:.2f}")
        print(f"Passed: {metric.score >= metric.threshold}")
        print(f"Reason: {metric.reason}")

        results.append((name, metric.score, metric.reason))

    return results


def evaluate_with_evaluate_function():
    """Demonstrate batch evaluation using evaluate() function."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION WITH evaluate() FUNCTION")
    print("=" * 70)

    metric = ArgumentCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
    )

    test_cases = [
        create_correct_args_test_case(),
        create_missing_args_test_case(),
        create_wrong_args_test_case(),
    ]

    result = evaluate(test_cases=test_cases, metrics=[metric])

    print(f"\nBatch evaluation completed for {len(test_cases)} test cases")

    return result


def demonstrate_tool_description_importance():
    """Show how tool descriptions affect evaluation quality."""
    print("\n" + "=" * 70)
    print("IMPACT OF TOOL DESCRIPTIONS ON EVALUATION")
    print("=" * 70)

    metric = ArgumentCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
    )

    # Same scenario with good vs poor tool description
    user_input = "Find Italian restaurants near Times Square"

    # Test case with good description
    good_description_case = LLMTestCase(
        input=user_input,
        actual_output="Here are Italian restaurants near Times Square...",
        tools_called=[
            ToolCall(
                name="RestaurantSearch",
                description="Search for restaurants by cuisine type and location. "
                "Accepts cuisine (string), location (string or coordinates), "
                "and optional filters like price_range and rating.",
                input_parameters={
                    "cuisine": "Italian",
                    "location": "Times Square, New York",
                },
            ),
        ],
    )

    # Test case with poor description
    poor_description_case = LLMTestCase(
        input=user_input,
        actual_output="Here are Italian restaurants near Times Square...",
        tools_called=[
            ToolCall(
                name="search",
                description="search stuff",  # Vague description
                input_parameters={
                    "cuisine": "Italian",
                    "location": "Times Square, New York",
                },
            ),
        ],
    )

    print("\n--- With Good Tool Description ---")
    metric.measure(good_description_case)
    print(f"Score: {metric.score:.2f}")
    print(f"Reason: {metric.reason}")

    print("\n--- With Poor Tool Description ---")
    metric.measure(poor_description_case)
    print(f"Score: {metric.score:.2f}")
    print(f"Reason: {metric.reason}")

    print("\nNote: Clear tool descriptions help the LLM evaluator understand")
    print("what arguments are expected and evaluate correctness more accurately.")


def compare_argument_quality():
    """Compare different levels of argument quality for same query."""
    print("\n" + "=" * 70)
    print("ARGUMENT QUALITY COMPARISON")
    print("=" * 70)

    metric = ArgumentCorrectnessMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
    )

    user_input = "Search for recent news about electric vehicles in 2024"

    # Excellent arguments
    excellent = LLMTestCase(
        input=user_input,
        actual_output="Here are recent EV news articles...",
        tools_called=[
            ToolCall(
                name="NewsSearch",
                description="Search news articles by topic, date range, and source.",
                input_parameters={
                    "query": "electric vehicles news",
                    "year": 2024,
                    "sort_by": "date",
                    "category": "automotive",
                },
            ),
        ],
    )

    # Good arguments
    good = LLMTestCase(
        input=user_input,
        actual_output="Here are recent EV news articles...",
        tools_called=[
            ToolCall(
                name="NewsSearch",
                description="Search news articles by topic, date range, and source.",
                input_parameters={
                    "query": "electric vehicles",
                    "year": 2024,
                },
            ),
        ],
    )

    # Poor arguments
    poor = LLMTestCase(
        input=user_input,
        actual_output="Here are recent EV news articles...",
        tools_called=[
            ToolCall(
                name="NewsSearch",
                description="Search news articles by topic, date range, and source.",
                input_parameters={
                    "query": "cars",  # Too vague
                    # Missing year filter
                },
            ),
        ],
    )

    cases = [
        ("Excellent", excellent),
        ("Good", good),
        ("Poor", poor),
    ]

    for name, case in cases:
        metric.measure(case)
        print(f"\n{name} Arguments:")
        print(f"  Input: {case.tools_called[0].input_parameters}")
        print(f"  Score: {metric.score:.2f}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("ARGUMENT CORRECTNESS METRIC DEMONSTRATION")
    print("Research Assistant Agent Example")
    print("=" * 70)

    print(
        """
    The Argument Correctness metric evaluates whether tool arguments
    correctly capture user intent. It's a REFERENCELESS metric - no
    expected arguments needed, just the user's input and tools_called.
    
    Key evaluation criteria:
    - Relevance: Do arguments relate to user's request?
    - Completeness: Are all necessary parameters provided?
    - Accuracy: Are values correct for the context?
    - Format: Are arguments in expected format/type?
    """
    )

    # Run individual evaluations
    results = evaluate_individual_cases()

    # Show impact of descriptions
    demonstrate_tool_description_importance()

    # Compare quality levels
    compare_argument_quality()

    # Batch evaluation
    evaluate_with_evaluate_function()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    for name, score, reason in results:
        status = "PASS" if score >= 0.7 else "FAIL"
        print(f"\n{name}:")
        print(f"  Score: {score:.2f} [{status}]")
        print(f"  Reason: {reason[:100]}...")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Argument Correctness is REFERENCELESS - no expected_tools needed
    2. The metric evaluates HOW tools are called, not WHICH tools
    3. Tool descriptions are crucial for accurate evaluation
    4. Score = Correct Arguments / Total Tool Calls
    5. Missing arguments reduce score (incomplete extraction)
    6. Wrong arguments reduce score (incorrect interpretation)
    7. Use with Tool Correctness for comprehensive tool evaluation
    """
    )


if __name__ == "__main__":
    main()
