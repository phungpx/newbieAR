"""
Step Efficiency Metric Demonstration

This script demonstrates the Step Efficiency metric by comparing efficient vs inefficient
execution paths for the same task. The metric evaluates whether the agent's execution
steps were necessary and optimal, penalizing redundant or unnecessary actions.

Key Concepts:
- Trace-based metric using @observe decorator
- Uses update_current_trace() to capture execution details
- Compares task requirements against actual execution steps
- Penalizes redundant, unnecessary, or inefficient actions

Reference: https://deepeval.com/docs/metrics-step-efficiency
"""

from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import ToolCall
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
# Simulated Tools
# =============================================================================


def get_weather(location: str) -> dict:
    """Get weather for a specific location."""
    return {
        "location": location,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45,
    }


def search_cities(region: str) -> list:
    """Search for cities in a region."""
    return ["San Francisco", "Los Angeles", "San Diego", "Sacramento"]


def get_coordinates(city: str) -> dict:
    """Get coordinates for a city."""
    return {"city": city, "lat": 37.7749, "lon": -122.4194}


def validate_location(location: str) -> bool:
    """Validate if a location exists."""
    return True


def format_weather_response(data: dict) -> str:
    """Format weather data into a readable response."""
    return (
        f"Weather in {data['location']}: {data['temperature']}°F, {data['condition']}"
    )


def search_web(query: str) -> list:
    """Search the web for information."""
    return [f"Result 1 for {query}", f"Result 2 for {query}"]


def filter_results(results: list, criteria: str) -> list:
    """Filter search results."""
    return results[:1]


def summarize_content(content: list) -> str:
    """Summarize a list of content."""
    return f"Summary of {len(content)} items"


# =============================================================================
# Agent Implementations - Efficient vs Inefficient
# =============================================================================


@observe
def efficient_weather_agent(user_input: str) -> str:
    """
    EFFICIENT implementation - Direct approach to get weather.
    Single tool call achieves the goal.
    Expected: HIGH efficiency score
    """
    tools_called = []

    # Direct weather lookup - single step
    weather = get_weather("San Francisco")
    tools_called.append(
        ToolCall(
            name="GetWeather",
            description="Gets current weather for a location",
            input={"location": "San Francisco"},
        )
    )

    output = f"The weather in San Francisco is {weather['temperature']}°F and {weather['condition']}."

    update_current_trace(input=user_input, output=output, tools_called=tools_called)

    return output


@observe
def inefficient_weather_agent(user_input: str) -> str:
    """
    INEFFICIENT implementation - Takes unnecessary steps to get weather.
    Multiple redundant tool calls for a simple task.
    Expected: LOW efficiency score
    """
    tools_called = []

    # Step 1: Unnecessary - Search for cities in California
    cities = search_cities("California")
    tools_called.append(
        ToolCall(
            name="SearchCities",
            description="Search for cities in a region",
            input={"region": "California"},
        )
    )

    # Step 2: Unnecessary - Get coordinates (not needed for weather)
    coords = get_coordinates("San Francisco")
    tools_called.append(
        ToolCall(
            name="GetCoordinates",
            description="Get geographic coordinates for a city",
            input={"city": "San Francisco"},
        )
    )

    # Step 3: Unnecessary - Validate location (already know it exists)
    valid = validate_location("San Francisco")
    tools_called.append(
        ToolCall(
            name="ValidateLocation",
            description="Validate if a location exists",
            input={"location": "San Francisco"},
        )
    )

    # Step 4: Finally get weather (the only needed step)
    weather = get_weather("San Francisco")
    tools_called.append(
        ToolCall(
            name="GetWeather",
            description="Gets current weather for a location",
            input={"location": "San Francisco"},
        )
    )

    # Step 5: Unnecessary - Format response (could be done inline)
    formatted = format_weather_response(weather)
    tools_called.append(
        ToolCall(
            name="FormatWeatherResponse",
            description="Format weather data into readable text",
            input={"data": weather},
        )
    )

    output = formatted

    update_current_trace(input=user_input, output=output, tools_called=tools_called)

    return output


@observe
def redundant_search_agent(user_input: str) -> str:
    """
    INEFFICIENT - Performs redundant similar searches.
    Expected: LOW efficiency score due to repetition
    """
    tools_called = []

    # Redundant Step 1: Vague search
    results1 = search_web("Python")
    tools_called.append(
        ToolCall(
            name="WebSearch",
            description="Search the web",
            input={"query": "Python"},
        )
    )

    # Redundant Step 2: Slightly more specific (should have done this first)
    results2 = search_web("Python programming")
    tools_called.append(
        ToolCall(
            name="WebSearch",
            description="Search the web",
            input={"query": "Python programming"},
        )
    )

    # Redundant Step 3: Even more specific (repetitive refinement)
    results3 = search_web("Python programming tutorials 2024")
    tools_called.append(
        ToolCall(
            name="WebSearch",
            description="Search the web",
            input={"query": "Python programming tutorials 2024"},
        )
    )

    # Step 4: Filter (could have searched correctly from start)
    filtered = filter_results(results3, "beginner")
    tools_called.append(
        ToolCall(
            name="FilterResults",
            description="Filter search results",
            input={"results": results3, "criteria": "beginner"},
        )
    )

    output = f"Found tutorials: {filtered}"

    update_current_trace(input=user_input, output=output, tools_called=tools_called)

    return output


@observe
def efficient_search_agent(user_input: str) -> str:
    """
    EFFICIENT - Direct, specific search from the start.
    Expected: HIGH efficiency score
    """
    tools_called = []

    # Single, well-crafted search
    results = search_web("Python programming tutorials for beginners 2024")
    tools_called.append(
        ToolCall(
            name="WebSearch",
            description="Search the web",
            input={"query": "Python programming tutorials for beginners 2024"},
        )
    )

    output = f"Found tutorials: {results}"

    update_current_trace(input=user_input, output=output, tools_called=tools_called)

    return output


@observe
def moderately_efficient_agent(user_input: str) -> str:
    """
    MODERATELY EFFICIENT - Some necessary steps, minor inefficiency.
    Expected: MEDIUM efficiency score
    """
    tools_called = []

    # Step 1: Search - Necessary
    results = search_web("Python tutorials 2024")
    tools_called.append(
        ToolCall(
            name="WebSearch",
            description="Search the web",
            input={"query": "Python tutorials 2024"},
        )
    )

    # Step 2: Filter - Necessary for refinement
    filtered = filter_results(results, "beginner")
    tools_called.append(
        ToolCall(
            name="FilterResults",
            description="Filter search results",
            input={"results": results, "criteria": "beginner"},
        )
    )

    # Step 3: Summarize - Arguably necessary for good output
    summary = summarize_content(filtered)
    tools_called.append(
        ToolCall(
            name="SummarizeContent",
            description="Summarize content",
            input={"content": filtered},
        )
    )

    output = f"Summary: {summary}"

    update_current_trace(input=user_input, output=output, tools_called=tools_called)

    return output


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_weather_agents():
    """Compare efficient vs inefficient weather agents."""
    print("\n" + "=" * 70)
    print("WEATHER AGENT COMPARISON: Efficient vs Inefficient")
    print("=" * 70)

    metric = StepEfficiencyMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        verbose_mode=True,
    )

    user_input = "What's the weather like in San Francisco?"

    agents = [
        ("Efficient Weather Agent (1 step)", efficient_weather_agent),
        ("Inefficient Weather Agent (5 steps)", inefficient_weather_agent),
    ]

    results = []

    for name, agent_func in agents:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")

        dataset = EvaluationDataset(goldens=[Golden(input=user_input)])

        for golden in dataset.evals_iterator(metrics=[metric]):
            output = agent_func(golden.input)
            print(f"\nUser Input: {golden.input}")
            print(f"Output: {output}")
            print(f"\n--- Efficiency Evaluation ---")
            print(f"Score: {metric.score:.2f}")
            print(f"Passed: {metric.score >= metric.threshold}")
            print(f"Reason: {metric.reason}")

            results.append((name, metric.score))

    return results


def evaluate_search_agents():
    """Compare efficient vs redundant search patterns."""
    print("\n" + "=" * 70)
    print("SEARCH AGENT COMPARISON: Direct vs Redundant Search")
    print("=" * 70)

    metric = StepEfficiencyMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        verbose_mode=True,
    )

    user_input = "Find Python programming tutorials for beginners"

    agents = [
        ("Efficient Search (1 step)", efficient_search_agent),
        ("Redundant Search (4 steps)", redundant_search_agent),
        ("Moderately Efficient (3 steps)", moderately_efficient_agent),
    ]

    results = []

    for name, agent_func in agents:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")

        dataset = EvaluationDataset(goldens=[Golden(input=user_input)])

        for golden in dataset.evals_iterator(metrics=[metric]):
            output = agent_func(golden.input)
            print(f"\nUser Input: {golden.input}")
            print(f"Output (truncated): {output[:100]}...")
            print(f"\n--- Efficiency Evaluation ---")
            print(f"Score: {metric.score:.2f}")
            print(f"Passed: {metric.score >= metric.threshold}")
            print(f"Reason: {metric.reason}")

            results.append((name, metric.score))

    return results


def demonstrate_step_counting():
    """Show how step count affects efficiency."""
    print("\n" + "=" * 70)
    print("STEP COUNT ANALYSIS")
    print("=" * 70)

    print(
        """
    Step Efficiency penalizes unnecessary steps. Here's why each agent's
    step count matters:

    WEATHER TASK: "What's the weather in San Francisco?"
    
    Efficient Agent (1 step):
    ├── GetWeather("San Francisco") → Done
    └── Score: ~1.0 (optimal path)
    
    Inefficient Agent (5 steps):
    ├── SearchCities("California")     → Unnecessary
    ├── GetCoordinates("SF")           → Unnecessary  
    ├── ValidateLocation("SF")         → Unnecessary
    ├── GetWeather("SF")               → Necessary
    └── FormatWeatherResponse(data)    → Unnecessary
    └── Score: ~0.2-0.4 (many unnecessary steps)
    
    SEARCH TASK: "Find Python tutorials"
    
    Efficient Agent (1 step):
    ├── WebSearch("Python tutorials 2024") → Done
    └── Score: ~1.0 (direct search)
    
    Redundant Agent (4 steps):
    ├── WebSearch("Python")                    → Too vague
    ├── WebSearch("Python programming")        → Redundant refinement
    ├── WebSearch("Python programming 2024")   → Should have done first
    └── FilterResults(results)                 → Could avoid with better search
    └── Score: ~0.3-0.5 (redundant searches)
    """
    )


def run_comprehensive_evaluation():
    """Run all evaluation scenarios."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE STEP EFFICIENCY EVALUATION")
    print("=" * 70)

    all_results = []

    # Weather agents comparison
    weather_results = evaluate_weather_agents()
    all_results.extend(weather_results)

    # Search agents comparison
    search_results = evaluate_search_agents()
    all_results.extend(search_results)

    # Step counting demonstration
    demonstrate_step_counting()

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("STEP EFFICIENCY METRIC DEMONSTRATION")
    print("Comparing Efficient vs Inefficient Agent Execution Paths")
    print("=" * 70)

    print(
        """
    The Step Efficiency metric evaluates whether an agent's execution
    was optimal for the given task. It penalizes:
    
    - Redundant steps (doing the same thing multiple times)
    - Unnecessary steps (steps that don't contribute to the goal)
    - Suboptimal paths (roundabout approaches to simple tasks)
    
    This demo compares agents with different efficiency levels:
    - Efficient agents: Direct, minimal steps
    - Inefficient agents: Redundant, unnecessary steps
    - Moderate agents: Some extra steps but mostly necessary
    """
    )

    # Run all evaluations
    all_results = run_comprehensive_evaluation()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for name, score in all_results:
        status = "PASS" if score >= 0.7 else "FAIL"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{name:40} [{bar}] {score:.2f} [{status}]")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. Step Efficiency is a TRACE-BASED metric (requires @observe)
    2. It evaluates NECESSITY of each step, not just task completion
    3. Efficient agents use MINIMAL steps to achieve the goal
    4. Redundant searches/lookups are heavily penalized
    5. The metric helps identify optimization opportunities
    6. Use with Task Completion to ensure efficiency doesn't hurt success
    
    Best Practices for High Efficiency Scores:
    - Design tools that accomplish goals in single calls
    - Avoid iterative refinement when direct approach works
    - Cache intermediate results to avoid repeated lookups
    - Combine frequently sequential operations into composite tools
    """
    )


if __name__ == "__main__":
    main()
