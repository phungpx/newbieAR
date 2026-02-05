# Conversational Q&A Agent with Memory Design

**Date:** 2026-02-04
**Status:** Approved

## Overview

A LangGraph-based conversational Q&A agent that uses BasicRAG for document retrieval and Graphiti for memory management. The agent intelligently routes queries, deciding whether to retrieve documents or answer from memory/general knowledge.

## Architecture & Components

### Core Components

1. **BasicRAG Tool** - Wrapped as a LangGraph-compatible tool for document retrieval
2. **Graphiti Memory** - Manages episodic memories and optionally extracts facts
3. **LLM Router** - Decides whether document retrieval is needed
4. **LLM Generator** - Produces final responses using available context

### State Structure

```python
class AgentState(TypedDict):
    user_id: str
    session_id: str
    messages: list[dict]  # Conversation history
    memory_context: str   # Retrieved memories from Graphiti
    documents_context: str  # Retrieved documents from BasicRAG
    needs_retrieval: bool  # Router decision
    final_response: str
```

### Configuration System

The agent accepts a config object specifying:
- Memory mode: `"session"`, `"cross_session"`, or `"user_scoped"`
- Memory type: `"episodic"` or `"episodic_with_facts"`
- BasicRAG collection name
- LLM settings for routing and generation

All components are injected as dependencies, making the agent testable and flexible.

## Memory System

### Memory Modes

1. **`"session"`** - Memory isolated per session. When a new session starts, previous memories aren't retrieved. Memory is saved with session context but only retrieved for matching session_id.

2. **`"cross_session"`** - All sessions for a user are connected. Graphiti retrieves memories across all sessions for the user_id, enabling continuity like "you asked about this last week."

3. **`"user_scoped"`** - Hybrid approach where each session has its own episodic memory, but extracted facts are shared across sessions. Good for remembering user preferences without cluttering context with old conversations.

### Memory Types

1. **`"episodic"`** - Stores raw conversation turns as episodic memories in Graphiti. Fast, simple, just tracks what was said.

2. **`"episodic_with_facts"`** - Stores conversations AND calls Graphiti's fact extraction to build entity/relationship graph. For example: extracts "User works at Google" as a persistent fact, separate from the episodic "User said: I work at Google."

### Implementation

Memory operations happen in dedicated nodes:
- **memory_retrieval node**: Queries Graphiti using user_id/session_id based on mode, retrieves relevant context
- **memory_save node**: Adds conversation turn to Graphiti after response generation, optionally triggers fact extraction

Graphiti handles the graph storage, temporal indexing, and semantic search automatically.

## Agent Flow & Nodes

### LangGraph Nodes

1. **`entry`** - Receives user query, initializes state with user_id, session_id, and message

2. **`retrieve_memory`** - Queries Graphiti for relevant memories based on current query and memory mode. Populates `memory_context` in state.

3. **`routing_decision`** - LLM analyzes query + memory_context to decide if document retrieval is needed. Sets `needs_retrieval` flag. Prompt examples:
   - "What did I ask last time?" → needs_retrieval=False
   - "Explain quantum computing from the docs" → needs_retrieval=True

4. **`retrieve_documents`** - Calls BasicRAG.retrieve() if needed, populates `documents_context`

5. **`generate_response`** - LLM generates final answer using query + memory_context + documents_context (if available)

6. **`save_memory`** - Adds conversation turn to Graphiti (user message + assistant response), triggers fact extraction if configured

7. **`end`** - Returns final response

### Control Flow

```
entry → retrieve_memory → routing_decision
                              ↓
                    [needs_retrieval?]
                    ↙              ↘
            retrieve_documents    (skip)
                    ↓              ↓
                  generate_response
                         ↓
                   save_memory → end
```

The conditional edge after `routing_decision` is the key to smart routing.

## Configuration & Initialization

### Configuration Structure

```python
@dataclass
class AgentConfig:
    # Memory settings
    memory_mode: Literal["session", "cross_session", "user_scoped"]
    memory_type: Literal["episodic", "episodic_with_facts"]

    # BasicRAG settings
    qdrant_collection_name: str
    retrieval_top_k: int = 5

    # LLM settings (uses existing settings from project)
    router_temperature: float = 0.0
    generator_temperature: float = 0.3

    # Optional: Override LLM clients
    custom_llm_client: Optional[OpenAILLMClient] = None
    custom_graphiti_client: Optional[Graphiti] = None
    custom_rag: Optional[BasicRAG] = None
```

### Agent Initialization

```python
# Create agent
agent = ConversationalAgent(config=AgentConfig(
    memory_mode="cross_session",
    memory_type="episodic_with_facts",
    qdrant_collection_name="my_docs"
))

# Run conversation
response = await agent.run(
    user_id="user_123",
    session_id="session_456",
    query="What did the docs say about authentication?"
)
```

### Dependency Injection

The agent internally creates:
- `BasicRAG` instance using project settings
- `GraphitiClient` and initializes Graphiti
- LLM clients for routing and generation

All can be overridden via config for testing or custom setups. The agent handles async initialization of Graphiti (build_indices_and_constraints) in its setup method.

## Error Handling & Integration

### Error Handling (Fail-Fast)

The agent raises exceptions immediately on failures, letting callers decide retry/fallback strategy:

- **Memory retrieval fails**: Raises `MemoryRetrievalError` - caller can retry or proceed without memory
- **Document retrieval fails**: Raises `DocumentRetrievalError` - BasicRAG connection issues
- **LLM failures**: Raises `LLMError` - routing or generation failures
- **Memory save fails**: Raises `MemorySaveError` - logs warning but doesn't block response return

All errors inherit from `AgentError` base class for easy catching. Each error includes context (user_id, session_id, operation) for debugging.

### Integration Points

```python
# Basic usage
agent = ConversationalAgent(config)
response = await agent.run(user_id="u1", session_id="s1", query="...")

# Batch processing
for query in queries:
    try:
        response = await agent.run(user_id, session_id, query)
    except DocumentRetrievalError:
        # Fallback: answer without retrieval
        pass

# Cleanup
await agent.cleanup()  # Closes Graphiti driver
```

### Observability

The agent logs key events using loguru (already in project):
- Memory retrieval results (# memories found)
- Routing decisions (needs_retrieval=True/False with reasoning)
- Document retrieval results (# docs retrieved)
- Response generation
- Memory save operations

LangGraph's built-in checkpointing can be enabled for debugging state transitions.

## Implementation Checklist

- [ ] Add langgraph dependency to pyproject.toml
- [ ] Create agent configuration dataclass
- [ ] Implement custom exception classes
- [ ] Create LangGraph state definition
- [ ] Implement memory retrieval node
- [ ] Implement routing decision node with LLM
- [ ] Implement document retrieval node (BasicRAG wrapper)
- [ ] Implement response generation node
- [ ] Implement memory save node
- [ ] Wire up LangGraph with conditional edges
- [ ] Create ConversationalAgent main class
- [ ] Add async initialization and cleanup methods
- [ ] Write unit tests for each node
- [ ] Write integration tests for full flows
- [ ] Add example usage script
