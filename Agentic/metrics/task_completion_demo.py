"""
Task Completion Metric Demonstration

This script demonstrates the Task Completion metric using a trip planner agent.
The metric evaluates whether the agent successfully accomplishes the task it was given
by extracting the task from the execution trace and comparing it with the actual outcome.

Key Concepts:
- Uses @observe decorator for tracing agent execution
- Uses update_current_trace() to capture input, output, and tools_called
- Evaluates task completion using LLM-as-a-judge
- Demonstrates complete, partial, and failed task scenarios

Reference: https://deepeval.com/docs/metrics-task-completion
"""

from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
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
# Tool Definitions (Simulated)
# =============================================================================


def search_flights(origin: str, destination: str, date: str) -> dict:
    """Simulate searching for flights."""
    return {
        "flights": [
            {
                "airline": "United",
                "price": 450,
                "departure": "08:00",
                "arrival": "14:30",
            },
            {
                "airline": "Delta",
                "price": 520,
                "departure": "10:30",
                "arrival": "17:00",
            },
            {
                "airline": "American",
                "price": 480,
                "departure": "14:00",
                "arrival": "20:30",
            },
        ],
        "origin": origin,
        "destination": destination,
        "date": date,
    }


def search_hotels(city: str, check_in: str, check_out: str) -> dict:
    """Simulate searching for hotels."""
    return {
        "hotels": [
            {"name": "Park Hyatt", "price": 350, "rating": 4.8},
            {"name": "Marriott Downtown", "price": 220, "rating": 4.5},
            {"name": "Holiday Inn", "price": 150, "rating": 4.2},
        ],
        "city": city,
        "check_in": check_in,
        "check_out": check_out,
    }


def search_restaurants(city: str, cuisine: str = None) -> dict:
    """Simulate searching for restaurants."""
    restaurants = [
        {"name": "Sukiyabashi Jiro", "cuisine": "Sushi", "rating": 4.9, "price": "$$$"},
        {
            "name": "Narisawa",
            "cuisine": "French-Japanese",
            "rating": 4.8,
            "price": "$$$$",
        },
        {"name": "Ichiran Ramen", "cuisine": "Ramen", "rating": 4.6, "price": "$"},
        {"name": "Tonki", "cuisine": "Tonkatsu", "rating": 4.5, "price": "$$"},
    ]
    if cuisine:
        restaurants = [
            r for r in restaurants if cuisine.lower() in r["cuisine"].lower()
        ]
    return {"restaurants": restaurants, "city": city}


def generate_itinerary(trip_details: dict) -> str:
    """Generate a formatted itinerary."""
    return f"""
    ===== YOUR TOKYO TRIP ITINERARY =====
    
    FLIGHTS:
    - Outbound: {trip_details.get('flight', 'Not booked')}
    
    ACCOMMODATION:
    - Hotel: {trip_details.get('hotel', 'Not booked')}
    
    DINING RECOMMENDATIONS:
    {chr(10).join(f"  - {r}" for r in trip_details.get('restaurants', ['None']))}
    
    =====================================
    """


# =============================================================================
# Agent Implementations
# =============================================================================


@observe
def complete_trip_planner(user_request: str) -> str:
    """
    A trip planner agent that completes all aspects of trip planning.
    This demonstrates a COMPLETE task completion scenario.
    """
    tools_called = []

    # Step 1: Search for flights
    flights = search_flights("LAX", "Tokyo", "2024-04-15")
    tools_called.append(
        ToolCall(
            name="SearchFlights",
            description="Search for available flights between two cities",
            input={"origin": "LAX", "destination": "Tokyo", "date": "2024-04-15"},
        )
    )

    # Step 2: Search for hotels
    hotels = search_hotels("Tokyo", "2024-04-15", "2024-04-20")
    tools_called.append(
        ToolCall(
            name="SearchHotels",
            description="Search for hotels in a city",
            input={
                "city": "Tokyo",
                "check_in": "2024-04-15",
                "check_out": "2024-04-20",
            },
        )
    )

    # Step 3: Search for restaurants
    restaurants = search_restaurants("Tokyo")
    tools_called.append(
        ToolCall(
            name="SearchRestaurants",
            description="Search for restaurants in a city",
            input={"city": "Tokyo"},
        )
    )

    # Step 4: Generate itinerary
    trip_details = {
        "flight": f"{flights['flights'][0]['airline']} - ${flights['flights'][0]['price']}",
        "hotel": f"{hotels['hotels'][0]['name']} - ${hotels['hotels'][0]['price']}/night",
        "restaurants": [r["name"] for r in restaurants["restaurants"][:3]],
    }
    itinerary = generate_itinerary(trip_details)
    tools_called.append(
        ToolCall(
            name="GenerateItinerary",
            description="Generate a formatted trip itinerary",
            input={"trip_details": trip_details},
        )
    )

    output = f"""
    I've planned your complete Tokyo trip! Here's what I found:

    FLIGHTS: Found {len(flights['flights'])} options. Best deal: {flights['flights'][0]['airline']} 
    at ${flights['flights'][0]['price']} departing at {flights['flights'][0]['departure']}.

    HOTELS: Found {len(hotels['hotels'])} options. Recommended: {hotels['hotels'][0]['name']} 
    at ${hotels['hotels'][0]['price']}/night with a {hotels['hotels'][0]['rating']} rating.

    RESTAURANTS: Here are top dining spots:
    {chr(10).join(f"    - {r['name']} ({r['cuisine']}) - {r['price']}" for r in restaurants['restaurants'][:3])}

    {itinerary}
    """

    # Update trace with complete execution details
    update_current_trace(input=user_request, output=output, tools_called=tools_called)

    return output


@observe
def partial_trip_planner(user_request: str) -> str:
    """
    A trip planner agent that only partially completes the task.
    This demonstrates a PARTIAL task completion scenario.
    """
    tools_called = []

    # Only search for restaurants (missing flights and hotels)
    restaurants = search_restaurants("Tokyo")
    tools_called.append(
        ToolCall(
            name="SearchRestaurants",
            description="Search for restaurants in a city",
            input={"city": "Tokyo"},
        )
    )

    output = f"""
    Here are some restaurant recommendations for Tokyo:
    
    {chr(10).join(f"- {r['name']} ({r['cuisine']}) - Rating: {r['rating']} - Price: {r['price']}" 
                  for r in restaurants['restaurants'])}
    
    Note: I wasn't able to search for flights and hotels at this time.
    """

    update_current_trace(input=user_request, output=output, tools_called=tools_called)

    return output


@observe
def failed_trip_planner(user_request: str) -> str:
    """
    A trip planner agent that fails to complete the task.
    This demonstrates a FAILED task completion scenario.
    """
    tools_called = []

    # Agent doesn't use any tools and provides unhelpful response
    output = """
    I'm sorry, but I'm unable to help with trip planning at this time. 
    Please try again later or contact customer support.
    """

    update_current_trace(input=user_request, output=output, tools_called=tools_called)

    return output


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_single_scenario(scenario_name: str, agent_func, user_input: str):
    """Evaluate a single scenario and print results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(f"User Input: {user_input}")

    # Create metric
    metric = TaskCompletionMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        verbose_mode=True,
    )

    # Create dataset with single golden
    dataset = EvaluationDataset(goldens=[Golden(input=user_input)])

    # Evaluate using evals_iterator
    for golden in dataset.evals_iterator(metrics=[metric]):
        output = agent_func(golden.input)
        print(f"\nAgent Output (truncated): {output[:500]}...")
        print(f"\n--- Evaluation Results ---")
        print(f"Task Completion Score: {metric.score}")
        print(
            f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}"
        )
        print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("TASK COMPLETION METRIC DEMONSTRATION")
    print("Using Trip Planner Agent")
    print("=" * 70)

    user_request = "Plan a weekend trip to Tokyo including flights, hotels, and restaurant recommendations"

    results = {}

    # Scenario 1: Complete Task
    results["complete"] = evaluate_single_scenario(
        "COMPLETE TASK - Full Trip Planning",
        complete_trip_planner,
        user_request,
    )

    # Scenario 2: Partial Task
    results["partial"] = evaluate_single_scenario(
        "PARTIAL TASK - Only Restaurants",
        partial_trip_planner,
        user_request,
    )

    # Scenario 3: Failed Task
    results["failed"] = evaluate_single_scenario(
        "FAILED TASK - No Planning Done",
        failed_trip_planner,
        user_request,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Complete Task Score: {results['complete']:.2f}")
    print(f"Partial Task Score:  {results['partial']:.2f}")
    print(f"Failed Task Score:   {results['failed']:.2f}")
    print("\nExpected Results:")
    print("- Complete: High score (0.8-1.0) - All trip components provided")
    print("- Partial:  Medium score (0.3-0.6) - Only restaurants provided")
    print("- Failed:   Low score (0.0-0.2) - No helpful response")


def run_standalone_measurement():
    """Demonstrate standalone metric.measure() usage."""
    print("\n" + "=" * 70)
    print("STANDALONE MEASUREMENT EXAMPLE")
    print("=" * 70)

    metric = TaskCompletionMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
    )

    # Create a single evaluation using evals_iterator
    dataset = EvaluationDataset(
        goldens=[Golden(input="Find me a good sushi restaurant in Tokyo")]
    )

    for golden in dataset.evals_iterator(metrics=[metric]):
        # Run agent with tracing
        @observe
        def simple_agent(user_input: str) -> str:
            restaurants = search_restaurants("Tokyo", "Sushi")
            output = f"Found sushi restaurant: {restaurants['restaurants'][0]['name']}"
            update_current_trace(
                input=user_input,
                output=output,
                tools_called=[
                    ToolCall(
                        name="SearchRestaurants",
                        input={"city": "Tokyo", "cuisine": "Sushi"},
                    )
                ],
            )
            return output

        output = simple_agent(golden.input)
        print(f"Input: {golden.input}")
        print(f"Output: {output}")
        print(f"Score: {metric.score}")
        print(f"Reason: {metric.reason}")


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    print("Task Completion Metric - Trip Planner Demo")
    print("This demonstrates how DeepEval evaluates task completion.\n")

    # Run all scenarios
    run_all_scenarios()

    # Run standalone example
    run_standalone_measurement()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
