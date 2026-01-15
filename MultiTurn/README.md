# Multi-Turn Evaluation Metrics

This module contains documentation and demonstration scripts for DeepEval's Multi-Turn evaluation metrics. These metrics are designed to evaluate LLM chatbots and agents across multi-turn conversations.

## Overview

Multi-Turn metrics evaluate conversations using `ConversationalTestCase` with `Turn` objects, enabling assessment of chatbot and agent behavior across extended interactions.

## Metrics Categories

### Category 1: Chatbot Conversation Metrics (Referenceless)

| Metric | Purpose | Documentation | Demo |
|--------|---------|---------------|------|
| Turn Relevancy | Measures response relevance throughout conversation | [TURN_RELEVANCY.md](docs/TURN_RELEVANCY.md) | [turn_relevancy_demo.py](metrics/turn_relevancy_demo.py) |
| Role Adherence | Measures adherence to assigned chatbot role | [ROLE_ADHERENCE.md](docs/ROLE_ADHERENCE.md) | [role_adherence_demo.py](metrics/role_adherence_demo.py) |
| Knowledge Retention | Measures fact retention from conversation | [KNOWLEDGE_RETENTION.md](docs/KNOWLEDGE_RETENTION.md) | [knowledge_retention_demo.py](metrics/knowledge_retention_demo.py) |
| Conversation Completeness | Measures user intention satisfaction | [CONVERSATION_COMPLETENESS.md](docs/CONVERSATION_COMPLETENESS.md) | [conversation_completeness_demo.py](metrics/conversation_completeness_demo.py) |

### Category 2: Agentic Multi-Turn Metrics

| Metric | Purpose | Documentation | Demo |
|--------|---------|---------------|------|
| Goal Accuracy | Evaluates planning and execution to reach goals | [GOAL_ACCURACY.md](docs/GOAL_ACCURACY.md) | [goal_accuracy_demo.py](metrics/goal_accuracy_demo.py) |
| Tool Use | Evaluates tool selection and argument generation | [TOOL_USE.md](docs/TOOL_USE.md) | [tool_use_demo.py](metrics/tool_use_demo.py) |
| Topic Adherence | Evaluates if agent answers only relevant topics | [TOPIC_ADHERENCE.md](docs/TOPIC_ADHERENCE.md) | [topic_adherence_demo.py](metrics/topic_adherence_demo.py) |

### Category 3: RAG Multi-Turn Metrics

| Metric | Purpose | Documentation | Demo |
|--------|---------|---------------|------|
| Turn Faithfulness | Measures factual accuracy grounded in context | [TURN_FAITHFULNESS.md](docs/TURN_FAITHFULNESS.md) | [turn_faithfulness_demo.py](metrics/turn_faithfulness_demo.py) |
| Turn Contextual Precision | Measures if relevant nodes ranked higher | [TURN_CONTEXTUAL_PRECISION.md](docs/TURN_CONTEXTUAL_PRECISION.md) | [turn_contextual_precision_demo.py](metrics/turn_contextual_precision_demo.py) |
| Turn Contextual Recall | Measures if context supports expected outcome | [TURN_CONTEXTUAL_RECALL.md](docs/TURN_CONTEXTUAL_RECALL.md) | [turn_contextual_recall_demo.py](metrics/turn_contextual_recall_demo.py) |
| Turn Contextual Relevancy | Measures if context is relevant to input | [TURN_CONTEXTUAL_RELEVANCY.md](docs/TURN_CONTEXTUAL_RELEVANCY.md) | [turn_contextual_relevancy_demo.py](metrics/turn_contextual_relevancy_demo.py) |

## Quick Start

```python
from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# Create a conversational test case
convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the weather like today?"),
        Turn(role="assistant", content="It's sunny and 72°F today."),
        Turn(role="user", content="Should I bring an umbrella?"),
        Turn(role="assistant", content="No, you won't need an umbrella since it's sunny."),
    ]
)

# Create and run the metric
metric = TurnRelevancyMetric(threshold=0.5)
metric.measure(convo_test_case)

print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
```

## Running Demo Scripts

Each metric has a demonstration script that shows various scenarios:

```bash
# Run from the MultiTurn directory
cd MultiTurn

# Run a specific demo
python metrics/turn_relevancy_demo.py
python metrics/role_adherence_demo.py
# ... etc
```

## Configuration

Copy the `.env.example` file to `.env` and configure your settings:

```bash
cp .env.example .env
```

Required environment variables:
- `llm_api_key`: Your OpenAI API key (or compatible provider)
- `llm_model`: Model to use for evaluation (default: gpt-4o-mini)
- `llm_base_url`: Base URL for API (optional, for custom endpoints)

## References

- [DeepEval Multi-Turn Metrics Documentation](https://deepeval.com/docs/metrics-turn-relevancy)
- [ConversationalTestCase Documentation](https://deepeval.com/docs/evaluation-test-cases)
