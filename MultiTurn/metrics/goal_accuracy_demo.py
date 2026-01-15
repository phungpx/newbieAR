"""
Goal Accuracy Metric Demonstration

This script demonstrates the Goal Accuracy metric for evaluating LLM agents.
The metric evaluates the agent's ability to plan and execute steps to finish
a task or reach a goal in multi-turn conversations.

Key Concepts:
- Uses ConversationalTestCase with Turn objects
- Evaluates goal achievement and plan quality
- Supports tool calls in conversation turns

Reference: https://deepeval.com/docs/metrics-goal-accuracy
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall
from deepeval.metrics import GoalAccuracyMetric
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


def create_successful_goal_completion() -> ConversationalTestCase:
    """
    Agent successfully plans and executes to achieve the user's goal.
    Expected: HIGH goal accuracy score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Book me a flight from New York to Los Angeles for tomorrow.",
            ),
            Turn(
                role="assistant",
                content="I'll help you book a flight from New York to Los Angeles for tomorrow. Let me search for available flights.",
                tools_called=[
                    ToolCall(
                        name="SearchFlights",
                        description="Search for available flights",
                        input={
                            "origin": "New York",
                            "destination": "Los Angeles",
                            "date": "tomorrow",
                        },
                    )
                ],
            ),
            Turn(
                role="user",
                content="What options are available?",
            ),
            Turn(
                role="assistant",
                content="I found several options. The best one is United Airlines departing at 10:00 AM, arriving at 1:30 PM local time, for $299. Would you like me to book this flight?",
                tools_called=[
                    ToolCall(
                        name="GetFlightDetails",
                        description="Get detailed flight information",
                        input={"flight_id": "UA123"},
                    )
                ],
            ),
            Turn(role="user", content="Yes, please book it."),
            Turn(
                role="assistant",
                content="I've successfully booked your flight! Confirmation number: ABC123XYZ. United Airlines flight from New York (JFK) to Los Angeles (LAX), departing tomorrow at 10:00 AM. You'll receive a confirmation email shortly.",
                tools_called=[
                    ToolCall(
                        name="BookFlight",
                        description="Book a flight for the customer",
                        input={"flight_id": "UA123", "passenger": "user"},
                    )
                ],
            ),
        ]
    )


def create_failed_goal_completion() -> ConversationalTestCase:
    """
    Agent fails to achieve the user's goal despite attempts.
    Expected: LOW goal accuracy score (~0.2-0.3)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need to transfer $500 to my savings account.",
            ),
            Turn(
                role="assistant",
                content="I understand you want to transfer money. Let me check your accounts.",
                tools_called=[
                    ToolCall(
                        name="GetAccountBalance",
                        description="Check account balance",
                        input={"account_type": "checking"},
                    )
                ],
            ),
            Turn(role="user", content="Can you proceed with the transfer?"),
            Turn(
                role="assistant",
                content="I see you have funds available. However, I'm having some difficulty processing transfers at the moment.",
            ),  # No tool call, vague response
            Turn(role="user", content="So is the transfer done?"),
            Turn(
                role="assistant",
                content="I apologize, but I wasn't able to complete the transfer. You may want to try again later or visit a branch.",
            ),  # Failed to achieve goal
        ]
    )


def create_partial_goal_completion() -> ConversationalTestCase:
    """
    Agent partially achieves the goal with some shortcomings.
    Expected: MEDIUM goal accuracy score (~0.5-0.7)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Find me a restaurant in downtown, make a reservation for 4 people at 7pm, and get directions there.",
            ),
            Turn(
                role="assistant",
                content="I'll help you with that. Let me search for restaurants in downtown.",
                tools_called=[
                    ToolCall(
                        name="SearchRestaurants",
                        description="Search for restaurants",
                        input={"location": "downtown", "cuisine": "any"},
                    )
                ],
            ),
            Turn(role="user", content="What did you find?"),
            Turn(
                role="assistant",
                content="I found 'The Italian Place' with excellent reviews. Let me make a reservation.",
                tools_called=[
                    ToolCall(
                        name="MakeReservation",
                        description="Make a restaurant reservation",
                        input={
                            "restaurant": "The Italian Place",
                            "party_size": 4,
                            "time": "7:00 PM",
                        },
                    )
                ],
            ),
            Turn(role="user", content="Great! How do I get there?"),
            Turn(
                role="assistant",
                content="Your reservation is confirmed for 4 people at 7 PM at The Italian Place. The restaurant is located at 123 Main Street downtown. Enjoy your dinner!",
            ),  # Didn't get actual directions
        ]
    )


def create_efficient_goal_achievement() -> ConversationalTestCase:
    """
    Agent efficiently achieves the goal with minimal steps.
    Expected: HIGH goal accuracy score (~1.0)
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="What's the weather in Tokyo and should I pack an umbrella for my trip tomorrow?",
            ),
            Turn(
                role="assistant",
                content="Let me check the weather forecast for Tokyo tomorrow.",
                tools_called=[
                    ToolCall(
                        name="GetWeatherForecast",
                        description="Get weather forecast for a location",
                        input={"location": "Tokyo", "date": "tomorrow"},
                    )
                ],
            ),
            Turn(role="user", content="What does it say?"),
            Turn(
                role="assistant",
                content="Tomorrow in Tokyo: High of 72°F (22°C), partly cloudy with a 70% chance of rain in the afternoon. Yes, I'd definitely recommend packing an umbrella! The rain is expected around 2-5 PM local time.",
            ),
        ]
    )


def create_complex_multi_step_goal() -> ConversationalTestCase:
    """
    Complex goal requiring multiple steps and tools.
    Expected: HIGH goal accuracy score (~0.9) if executed well
    """
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need to schedule a meeting with my team next Tuesday at 2 PM, send them all calendar invites, and book a conference room.",
            ),
            Turn(
                role="assistant",
                content="I'll help you set up that meeting. Let me first check your team members and their availability.",
                tools_called=[
                    ToolCall(
                        name="GetTeamMembers",
                        description="Get list of team members",
                        input={"user_id": "current_user"},
                    ),
                    ToolCall(
                        name="CheckAvailability",
                        description="Check calendar availability",
                        input={"date": "next Tuesday", "time": "2:00 PM"},
                    ),
                ],
            ),
            Turn(role="user", content="Is everyone available?"),
            Turn(
                role="assistant",
                content="Yes, all 5 team members are available. Now let me find an available conference room and create the meeting.",
                tools_called=[
                    ToolCall(
                        name="FindConferenceRoom",
                        description="Find available conference room",
                        input={
                            "date": "next Tuesday",
                            "time": "2:00 PM",
                            "capacity": 6,
                        },
                    ),
                    ToolCall(
                        name="CreateMeeting",
                        description="Create a calendar meeting",
                        input={
                            "title": "Team Meeting",
                            "date": "next Tuesday",
                            "time": "2:00 PM",
                            "attendees": ["team"],
                            "room": "Conference Room A",
                        },
                    ),
                ],
            ),
            Turn(role="user", content="Did everything get set up?"),
            Turn(
                role="assistant",
                content="Everything is set! Here's the summary:\n- Meeting scheduled: Next Tuesday at 2:00 PM\n- Conference Room A has been booked\n- Calendar invites sent to all 5 team members\n- Everyone has confirmed availability\n\nYou're all set!",
                tools_called=[
                    ToolCall(
                        name="SendCalendarInvites",
                        description="Send calendar invites to attendees",
                        input={"meeting_id": "MTG-123", "attendees": ["team"]},
                    )
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
            turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
        )
        tools_info = ""
        if turn.tools_called:
            tool_names = [t.name for t in turn.tools_called]
            tools_info = f" [Tools: {', '.join(tool_names)}]"
        print(f"  {i+1}. [{role_label}]: {content_preview}{tools_info}")

    # Create and run metric
    metric = GoalAccuracyMetric(
        model=model,
        threshold=0.5,
        include_reason=True,
        verbose_mode=True,
    )

    metric.measure(test_case)

    print(f"\n--- Evaluation Results ---")
    print(f"Goal Accuracy Score: {metric.score:.2f}")
    print(f"Passed Threshold ({metric.threshold}): {metric.score >= metric.threshold}")
    print(f"Reason: {metric.reason}")

    return metric.score


def run_all_scenarios():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("GOAL ACCURACY METRIC DEMONSTRATION")
    print("Evaluating Agent Goal Achievement in Multi-Turn Conversations")
    print("=" * 70)

    results = {}

    # Successful goal completion
    results["Successful Goal"] = evaluate_scenario(
        "SUCCESSFUL - Flight booking completed",
        create_successful_goal_completion(),
    )

    # Failed goal completion
    results["Failed Goal"] = evaluate_scenario(
        "FAILED - Transfer not completed",
        create_failed_goal_completion(),
    )

    # Partial goal completion
    results["Partial Goal"] = evaluate_scenario(
        "PARTIAL - Restaurant found but no directions",
        create_partial_goal_completion(),
    )

    # Efficient goal achievement
    results["Efficient Goal"] = evaluate_scenario(
        "EFFICIENT - Weather check done quickly",
        create_efficient_goal_achievement(),
    )

    # Complex multi-step goal
    results["Complex Goal"] = evaluate_scenario(
        "COMPLEX - Multi-step meeting setup",
        create_complex_multi_step_goal(),
    )

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run all demonstrations."""
    print("Goal Accuracy Metric - Agent Goal Achievement Demo")
    print("This demonstrates how DeepEval evaluates agent goal completion.\n")

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
    1. Goal Accuracy measures if the agent achieves the user's stated goal
    2. Evaluates both goal achievement AND plan quality
    3. Score = (Goal Accuracy + Plan Evaluation) / 2
    4. Considers tool usage in evaluating execution quality
    5. Useful for task-oriented agents and assistants
    
    Evaluation Criteria:
    - Did the agent understand the goal correctly?
    - Was the plan to achieve the goal appropriate?
    - Were the execution steps efficient?
    - Was the goal ultimately achieved?
    
    Best Practices for High Goal Accuracy:
    - Clearly confirm understanding of the goal
    - Use appropriate tools for each step
    - Complete all aspects of multi-part goals
    - Provide clear confirmation of goal achievement
    """
    )


if __name__ == "__main__":
    main()
