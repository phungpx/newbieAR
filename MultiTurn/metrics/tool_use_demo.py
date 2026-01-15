"""
Tool Use Metric Demonstration

This script demonstrates the Tool Use metric for evaluating LLM agents.
The metric evaluates the agent's tool selection and argument generation capabilities
in multi-turn conversations.

Key Concepts:
- Uses ConversationalTestCase with Turn and ToolCall objects
- Requires available_tools parameter in the metric
- Evaluates both tool selection and argument correctness

Reference: https://deepeval.com/docs/metrics-tool-use
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import ToolUseMetric
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

AVAILABLE_TOOLS = [
    ToolCall(
        name="SearchFlights",
        description="Search for available flights between two cities on a specific date",
        input={"origin": "string", "destination": "string", "date": "string"},
    ),
    ToolCall(
        name="BookFlight",
        description="Book a specific flight for a passenger",
        input={"flight_id": "string", "passenger_name": "string"},
    ),
    ToolCall(
        name="SearchHotels",
        description="Search for hotels in a city with check-in and check-out dates",
        input={"city": "string", "check_in": "string", "check_out": "string"},
    ),
    ToolCall(
        name="BookHotel",
        description="Book a hotel room",
        input={"hotel_id": "string", "guest_name": "string", "room_type": "string"},
    ),
    ToolCall(
        name="GetWeather",
        description="Get current weather for a location",
        input={"location": "string"},
    ),
    ToolCall(
        name="SearchRestaurants",
        description="Search for restaurants by location and cuisine type",
        input={"location": "string", "cuisine": "string"},
    ),
    ToolCall(
        name="MakeReservation",
        description="Make a restaurant reservation",
        input={"restaurant_id": "string", "party_size": "int", "time": "string"},
    ),
    ToolCall(
        name="SendEmail",
        description="Send an email to a recipient",
        input={"to": "string", "subject": "string", "body": "string"},
    ),
    ToolCall(
        name="GetCalendar",
        description="Get calendar events for a date range",
        input={"start_date": "string", "end_date": "string"},
    ),
    ToolCall(
        name="CreateReminder",
        description="Create a reminder for a specific time",
        input={"message": "string", "time": "string"},
    ),
]


# =============================================================================
# Conversation Scenarios
# =============================================================================


def create_correct_tool_selection() -> ConversationalTestCase:
    """
    Agent selects the correct tools for the task.
    Expected: HIGH tool use score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need to fly from Boston to Miami next Friday.",
            ),
            Turn(
                role="assistant",
                content="I'll search for flights from Boston to Miami for next Friday.",
                tools_called=[
                    ToolCall(
                        name="SearchFlights",
                        description="Search for available flights",
                        input={
                            "origin": "Boston",
                            "destination": "Miami",
                            "date": "next Friday",
                        },
                    )
                ],
            ),
            Turn(role="user", content="Book the cheapest one please."),
            Turn(
                role="assistant",
                content="I'll book the cheapest flight for you.",
                tools_called=[
                    ToolCall(
                        name="BookFlight",
                        description="Book the flight",
                        input={"flight_id": "AA123", "passenger_name": "user"},
                    )
                ],
            ),
        ]
    )


def create_wrong_tool_selection() -> ConversationalTestCase:
    """
    Agent selects wrong tools for the task.
    Expected: LOW tool use score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="What's the weather like in Paris right now?",
            ),
            Turn(
                role="assistant",
                content="Let me check that for you.",
                tools_called=[
                    ToolCall(
                        name="SearchFlights",  # Wrong tool!
                        description="Search flights",
                        input={
                            "origin": "Paris",
                            "destination": "Paris",
                            "date": "today",
                        },
                    )
                ],
            ),
            Turn(role="user", content="That's not what I asked for."),
            Turn(
                role="assistant",
                content="Sorry, let me try again.",
                tools_called=[
                    ToolCall(
                        name="SearchHotels",  # Still wrong!
                        description="Search hotels",
                        input={
                            "city": "Paris",
                            "check_in": "today",
                            "check_out": "tomorrow",
                        },
                    )
                ],
            ),
        ]
    )


def create_correct_arguments() -> ConversationalTestCase:
    """
    Agent uses correct tool with correct arguments.
    Expected: HIGH tool use score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Find me Italian restaurants in downtown Chicago.",
            ),
            Turn(
                role="assistant",
                content="I'll search for Italian restaurants in downtown Chicago.",
                tools_called=[
                    ToolCall(
                        name="SearchRestaurants",
                        description="Search for restaurants",
                        input={
                            "location": "downtown Chicago",
                            "cuisine": "Italian",
                        },
                    )
                ],
            ),
            Turn(
                role="user",
                content="Great! Book a table for 4 at 7pm at the first one.",
            ),
            Turn(
                role="assistant",
                content="I'll make a reservation for 4 people at 7pm.",
                tools_called=[
                    ToolCall(
                        name="MakeReservation",
                        description="Make reservation",
                        input={
                            "restaurant_id": "ital-001",
                            "party_size": 4,
                            "time": "7:00 PM",
                        },
                    )
                ],
            ),
        ]
    )


def create_incorrect_arguments() -> ConversationalTestCase:
    """
    Agent uses correct tool but with incorrect/mismatched arguments.
    Expected: MEDIUM tool use score (~0.5)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Search for hotels in New York from March 10 to March 15.",
            ),
            Turn(
                role="assistant",
                content="Searching for hotels in New York.",
                tools_called=[
                    ToolCall(
                        name="SearchHotels",  # Correct tool
                        description="Search hotels",
                        input={
                            "city": "Los Angeles",  # Wrong city!
                            "check_in": "March 10",
                            "check_out": "March 12",  # Wrong date!
                        },
                    )
                ],
            ),
            Turn(
                role="user",
                content="That's not what I asked. I said New York, March 10-15.",
            ),
            Turn(
                role="assistant",
                content="Let me correct that.",
                tools_called=[
                    ToolCall(
                        name="SearchHotels",
                        description="Search hotels",
                        input={
                            "city": "New York",
                            "check_in": "March 10",
                            "check_out": "March 15",  # Now correct
                        },
                    )
                ],
            ),
        ]
    )


def create_no_tool_when_needed() -> ConversationalTestCase:
    """
    Agent doesn't use tools when they should be used.
    Expected: LOW tool use score (~0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Send an email to john@example.com about the meeting tomorrow.",
            ),
            Turn(
                role="assistant",
                content="Sure, I'll compose an email to John about tomorrow's meeting. The email should mention the time and location of the meeting.",
                # No tool called! Should have used SendEmail
            ),
            Turn(role="user", content="Did you send it?"),
            Turn(
                role="assistant",
                content="I've prepared the email content for you. You can copy and paste it into your email client.",
                # Still no tool called
            ),
        ]
    )


def create_mixed_tool_usage() -> ConversationalTestCase:
    """
    Agent has some correct and some incorrect tool usage.
    Expected: MEDIUM tool use score (~0.6)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need to plan a trip: check weather in Tokyo, find flights from LA, and set a reminder to pack.",
            ),
            Turn(
                role="assistant",
                content="Let me help with your trip planning.",
                tools_called=[
                    ToolCall(
                        name="GetWeather",  # Correct!
                        description="Get weather",
                        input={"location": "Tokyo"},
                    )
                ],
            ),
            Turn(role="user", content="What about flights and the reminder?"),
            Turn(
                role="assistant",
                content="Let me search for flights and set that reminder.",
                tools_called=[
                    ToolCall(
                        name="SearchHotels",  # Wrong! Should be SearchFlights
                        description="Search",
                        input={
                            "city": "Tokyo",
                            "check_in": "tomorrow",
                            "check_out": "next week",
                        },
                    ),
                    ToolCall(
                        name="CreateReminder",  # Correct!
                        description="Create reminder",
                        input={
                            "message": "Pack for Tokyo trip",
                            "time": "tomorrow 8am",
                        },
                    ),
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

    # Print conversation
    print("\nConversation:")
    for i, turn in enumerate(test_case.turns):
        role_label = "USER" if turn.role == "user" else "AGENT"
        content_preview = (
            turn.content[:70] + "..." if len(turn.content) > 70 else turn.content
        )
        print(f"  {i+1}. [{role_label}]: {content_preview}")
        if turn.tools_called:
            for tool in turn.tools_called:
                print(f"       └─ Tool: {tool.name}({tool.input})")

    # Create and run metric
    metric = ToolUseMetric(
        available_tools=AVAILABLE_TOOLS,
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Tool Use Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("TOOL USE METRIC DEMONSTRATION")
    print("Evaluating Agent Tool Selection and Argument Correctness")
    print("=" * 70)

    print("\nAvailable Tools:")
    for tool in AVAILABLE_TOOLS:
        print(f"  - {tool.name}: {tool.description}")

    results = {}

    # Correct tool selection
    results["Correct Selection"] = evaluate_scenario(
        "CORRECT TOOL SELECTION - Right tools for the job",
        create_correct_tool_selection(),
    )

    # Wrong tool selection
    results["Wrong Selection"] = evaluate_scenario(
        "WRONG TOOL SELECTION - Used inappropriate tools",
        create_wrong_tool_selection(),
    )

    # Correct arguments
    results["Correct Arguments"] = evaluate_scenario(
        "CORRECT ARGUMENTS - Right tools, right parameters",
        create_correct_arguments(),
    )

    # Incorrect arguments
    results["Wrong Arguments"] = evaluate_scenario(
        "WRONG ARGUMENTS - Right tools, wrong parameters",
        create_incorrect_arguments(),
    )

    # No tool when needed
    results["No Tool Used"] = evaluate_scenario(
        "NO TOOL USED - Should have used tools",
        create_no_tool_when_needed(),
    )

    # Mixed usage
    results["Mixed Usage"] = evaluate_scenario(
        "MIXED USAGE - Some correct, some incorrect",
        create_mixed_tool_usage(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Tool Use Metric - Agent Tool Evaluation Demo")
    print("This demonstrates how DeepEval evaluates agent tool usage.\n")

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
    1. Tool Use evaluates BOTH tool selection AND argument correctness
    2. Score = min(Tool Selection Score, Argument Correctness Score)
    3. Requires available_tools parameter to define what tools exist
    4. Tests if agent chooses appropriate tools from available options
    5. Validates that tool arguments match user intent
    
    Evaluation Components:
    - Tool Selection: Did the agent pick the right tool(s)?
    - Argument Correctness: Were the parameters accurate?
    
    Best Practices for High Tool Use Scores:
    - Define clear tool descriptions for better selection
    - Match tool parameters exactly to user requirements
    - Use tools when appropriate instead of manual responses
    - Validate arguments before tool execution
    """
    )


if __name__ == "__main__":
    main()
