# Building an Intelligent Dual-Agent Chatbot with LangGraph: Emotional vs Logical Routing

## Introduction

Have you ever wished your chatbot could understand the nuance of human interaction—knowing when to be empathetic and when to be purely logical? In this comprehensive guide, we'll build an intelligent dual-agent system using **LangGraph** that classifies user messages and routes them to either a compassionate therapist agent or a logical reasoning agent.

This project demonstrates the power of **agentic AI workflows**, where language models cooperate within a directed graph structure to provide contextually appropriate responses.

## What We're Building

A conversational AI system that:
- 🧠 **Classifies** incoming user messages as either emotional or logical
- 🎯 **Routes** messages to the appropriate agent based on classification
- 💬 **Empathizes** using a therapist agent for emotional messages
- 📊 **Analyzes** using a logical agent for factual/informational requests
- 🔄 **Maintains** conversation state throughout the dialogue

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  Classifier Agent           │
        │  (Emotional or Logical?)    │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │  Router Node                │
        │  (Decision Point)           │
        └──────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   ┌─────────┐           ┌─────────┐
   │Therapist│           │ Logical │
   │  Agent  │           │  Agent  │
   └────┬────┘           └────┬────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │  Response to User           │
        └─────────────────────────────┘
```

## Technology Stack

- **LangGraph**: Orchestrates the multi-agent workflow
- **LangChain**: Provides LLM abstractions and utilities
- **Ollama**: Runs local LLM (gemma:2b) for privacy
- **Pydantic**: Ensures structured output from LLM
- **Python 3.8+**: For async/type-safe implementation

## Step-by-Step Implementation

### Step 1: Import Dependencies

```python
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
```

These imports give us:
- **StateGraph**: The core of LangGraph for building directed workflows
- **add_messages**: A helper for managing conversation state
- **ChatOllama**: LangChain's interface to local Ollama models
- **BaseModel & Field**: Pydantic for structured outputs

### Step 2: Initialize the LLM

```python
# Initialize Ollama chat model (gemma:2b) running at the local Ollama server
llm = ChatOllama(model="gemma:2b", ollama_url="http://localhost:11434")

print("Initialized LLM:", llm)
```

We're using **gemma:2b**, a 2-billion parameter model perfect for:
- Running locally without GPU requirements
- Fast inference (suitable for chatbots)
- Lower latency responses

**Prerequisites:**
```bash
# Install Ollama from https://ollama.ai
ollama pull gemma:2b
ollama serve  # Start the server
```

### Step 3: Define Message Classification Schema

```python
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )
```

Pydantic's `BaseModel` ensures:
- Type-safe structured outputs from the LLM
- Validation of LLM responses
- Clear schema that the LLM can follow

### Step 4: Define Application State

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
```

The State class holds:
- **messages**: Conversation history (add_messages automatically manages updates)
- **message_type**: The classified type of the current message

### Step 5: Create the Classifier Node

```python
def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system", 
            "content": "You are a message classifier that determines if a message requires an 'emotional' or 'logical' response."
        },
        {"role": "user", "content": last_message.content}
    ])

    return {"message_type": result.message_type}
```

**Key Points:**
- `with_structured_output()` forces the LLM to return valid Pydantic models
- System prompt guides the LLM's classification behavior
- Returns updated state with classification result

### Step 6: Create the Router Node

```python
def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    
    return {"next": "logical"}
```

The router is a conditional node that:
- Reads the message classification
- Returns the next node to execute
- Defaults to "logical" for safety

### Step 7: Implement Therapist Agent

```python
def therapist_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. 
                     Focus on the emotional aspects of the user's message.
                     Show empathy, validate their feelings, and help them process their emotions.
                     Ask thoughtful questions to help them explore their feelings more deeply.
                     Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}
```

**Therapist Characteristics:**
- Empathetic and validating tone
- Focuses on emotional understanding
- Asks open-ended questions
- Provides emotional support

### Step 8: Implement Logical Agent

```python
def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. 
                     Focus only on facts and information.
                     Provide clear, concise answers based on logic and evidence.
                     Do not address emotions or provide emotional support.
                     Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}
```

**Logical Agent Characteristics:**
- Fact-based responses
- Evidence-driven reasoning
- Direct and concise communication
- No emotional interpretation

### Step 9: Build the Graph

```python
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

# Define edges
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

# Conditional routing based on message type
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)

# Both agents end the conversation
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()
```

**Graph Flow:**
1. START → classifier (analyze message)
2. classifier → router (decide agent)
3. router → therapist OR logical (route to appropriate agent)
4. Both agents → END (conclude conversation turn)

### Step 10: Run the Chatbot

```python
def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()
```

## Example Conversations

### Emotional Message Routing
```
User: I'm feeling really overwhelmed with work lately. 
      Everyone keeps expecting more from me.

[Classified as: emotional]
[Routed to: Therapist Agent]