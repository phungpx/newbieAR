# Agentic Evaluation

## [Argument Correctness](https://deepeval.com/docs/metrics-argument-correctness)

### Definition

The argument correctness metric is an agentic LLM metric that assesses your LLM agent's `ability to generate the correct arguments` for the tools it calls. It is calculated by determining whether the arguments for each tool call is correct based on the `input`.

### Required Arguments

To use the ArgumentCorrectnessMetric, you'll have to provide the following arguments when creating an LLMTestCase:

- `input`
- `actual_output`
- `tools_called`

### How Is It Calculated?

Argument Correctness = Total Number of Tool Calls / Number of Correctly Generated Input Parameters

### Example

[here](./metrics/argument_corectness.py)

## [Step Efficiency](https://deepeval.com/docs/metrics-step-efficiency)

### Definition

The Step Efficiency metric is an agentic metric that extracts the task from your agent's trace and evaluates the `efficiency of your agent's execution steps` in completing that task. It analyzes your agent's full trace to determine the task and execution efficiency.

### Required Arguments

The StepEfficiencyMetric is a trace-only metric and **MUST** be used with `evals_iterator` or `@observe` decorator. It requires:

- Setting up tracing with `@observe` decorators
- Using `update_current_trace()` to provide:
  - `input`
  - `output`
  - `tools_called`

### How Is It Calculated?

Step Efficiency Score = AlignmentScore(Task, Execution Steps)

The metric:
1. Extracts the **Task** from the trace
2. Evaluates the **agent's execution steps** from the trace
3. Uses an LLM to generate the final score, penalizing unnecessary actions

### Example

[here](./metrics/step_efficiency.py)
