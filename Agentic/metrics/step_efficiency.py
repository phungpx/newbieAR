from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.test_case import ToolCall
from deepeval.models.llms import LocalModel
from settings import ProjectSettings

settings = ProjectSettings()

# Initialize the LLM model
model = LocalModel(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=settings.llm_temperature,
)


# Simulate a tool call function
@observe
def tool_call(input: str) -> list[ToolCall]:
    """
    Simulates calling tools based on the input.
    In a real scenario, this would call actual tools.
    """
    # Example: For weather query, we need a weather tool
    if "weather" in input.lower():
        return [
            ToolCall(
                name="GetWeather",
                description="Gets current weather information for a location",
                input={"location": "SF"},
            )
        ]
    # Example: For search queries
    elif "search" in input.lower() or "find" in input.lower():
        return [
            ToolCall(
                name="WebSearch",
                description="Searches the web for information",
                input={"query": input},
            )
        ]
    else:
        return [
            ToolCall(
                name="GeneralQuery",
                description="Handles general queries",
                input={"query": input},
            )
        ]


# Simulate an LLM function
def llm(input: str, tools: list[ToolCall]) -> str:
    """
    Simulates an LLM response based on input and available tools.
    In a real scenario, this would call your actual LLM.
    """
    if "weather" in input.lower():
        return "The weather in San Francisco is currently 65°F and sunny."
    elif "search" in input.lower():
        return f"I found relevant information about: {input}"
    else:
        return f"I processed your query: {input}"


# Main agent function with tracing
@observe
def agent(input: str) -> str:
    """
    Main agent function that orchestrates tool calls and LLM responses.
    This function is traced by DeepEval.
    """
    # Get tools based on input
    tools = tool_call(input)

    # Get LLM response with tools
    output = llm(input, tools)

    # Update the current trace with input, output, and tools
    update_current_trace(input=input, output=output, tools_called=tools)

    return output


def main():
    """
    Main function to run the Step Efficiency evaluation.
    """
    # Create dataset with example queries
    dataset = EvaluationDataset(
        goldens=[
            Golden(input="What's the weather like in SF?"),
            Golden(input="Search for information about Python programming"),
            Golden(input="Tell me about artificial intelligence"),
        ]
    )

    # Initialize Step Efficiency metric
    metric = StepEfficiencyMetric(
        model=model,
        threshold=0.7,
        include_reason=True,
        verbose_mode=True,
    )

    # Iterate through dataset and evaluate
    print("Starting Step Efficiency Evaluation...")
    print("=" * 60)

    for golden in dataset.evals_iterator(metrics=[metric]):
        print(f"\nEvaluating: {golden.input}")
        result = agent(golden.input)
        print(f"Agent Output: {result}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")


if __name__ == "__main__":
    main()
