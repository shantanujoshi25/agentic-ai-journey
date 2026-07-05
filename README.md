# Agentic AI Lab

> Hands-on lab building production-minded AI agents with **AWS Bedrock**, **LangChain**, and **LangGraph** — from raw tool-use loops to stateful, checkpointed ReAct agents.

This is a structured, build-first learning journey. Each week ships working code plus a short write-up of the concepts behind it. The goal isn't to collect tutorials — it's to understand *what the abstractions are actually doing* so I can reason about agent behavior, cost, and failure modes in production rather than treating any framework as magic.

The applied domain throughout is **payments / accounts payable** (transaction lookups, vendor search, currency conversion), which keeps the exercises grounded in a realistic enterprise workflow instead of toy examples.

---

## Stack

| Layer | Tool |
|-------|------|
| Model + runtime | AWS Bedrock (Converse API) |
| Models | Claude Sonnet & Haiku via Bedrock inference profiles |
| Agent orchestration | LangGraph |
| Model bindings | `langchain-aws` (`ChatBedrockConverse`) |
| SDK | `boto3`, `anthropic` |
| Language | Python 3.11+ |

---

## Repo structure

```
.
├── week-01-foundations/
│   ├── 01_hello_bedrock.py      # Converse API call + token/cost tracking
│   ├── 02_chat.py               # Multi-turn conversation with history
│   ├── 03_tool_use.py           # Raw ReAct loop, two mock tools
│   └── notes.md
├── week-02-langgraph/
│   ├── 01_state_graph_basics.py # StateGraph from scratch: state, nodes, edges
│   ├── 02_react_agent.py        # create_react_agent on Bedrock
│   ├── 03_streaming.py          # Streaming steps and tokens
│   ├── 04_checkpointing.py      # SqliteSaver + multi-turn threads
│   ├── 05_human_in_the_loop.py  # interrupt() for approval gates
│   └── notes.md
├── .env.example
└── README.md
```

---

## Week 1 — Foundations

**Goal:** understand the primitive that every agent framework is built on, before touching a framework.

Before you can appreciate what LangGraph gives you, you need to build an agent loop by hand. Week 1 does exactly that: it walks the model API from a single call up to a working tool-use loop, so the "agent runtime" stops being a black box.

| Module | What it covers |
|--------|----------------|
| `01_hello_bedrock.py` | A single Bedrock Converse call to Claude. Logs input/output token counts and estimates per-call cost from turn one. |
| `02_chat.py` | Multi-turn conversation. Shows why input tokens grow every turn — you resend the full history each time — and why context pruning matters for cost and latency. |
| `03_tool_use.py` | A **raw ReAct loop** built by hand with two mock tools (calculator, weather). Reason → Act → Observe → repeat, driven entirely by the model's `stop_reason` and `tool_use` blocks. This is the loop LangGraph abstracts. |

**Concepts locked in:**
- Foundation models vs. instruction-tuned; tool use / function calling as a structured emit-run-feedback cycle
- The ReAct loop and why the model is *not* the runtime — your code is
- Context windows, token growth, and cost-per-turn awareness
- Stop reasons (`end_turn` / `tool_use` / `max_tokens`) as the control signal
- A **provider abstraction** so the same agent code can target Bedrock Converse or a direct model API — swapping the backend without rewriting the loop

**Production discipline baked in from day one:**
- Always log token counts
- Always set a hard iteration limit on any agent loop (a runaway loop is a runaway bill)
- Check the cost dashboard as a reflex

---

## Week 2 — LangGraph Fundamentals

**Goal:** rebuild the Week 1 loop with LangGraph, and see with my own eyes that a "fancy" agent still boils down to the raw tool-use loop underneath.

The mental frame for the week: **LangGraph is not a magic agent library — it's a graph orchestration library with agent conveniences on top.** Every agent, no matter how elaborate, reduces to nodes, edges, and shared state driving a tool-use cycle.

| Module | What it covers |
|--------|----------------|
| `01_state_graph_basics.py` | A `StateGraph` built from scratch — defining state, adding nodes and edges — with no prebuilt agent helper, to make the graph model concrete. |
| `02_react_agent.py` | `create_react_agent` wired to Claude on Bedrock via `ChatBedrockConverse`, then a look under the hood at what it generates. |
| `03_streaming.py` | Streaming intermediate steps and tokens, so the agent's reasoning is observable rather than a black-box final answer. |
| `04_checkpointing.py` | Durable state with `SqliteSaver` and thread IDs — multi-turn memory that survives across invocations. |
| `05_human_in_the_loop.py` | `interrupt()` to pause the graph for human approval before a consequential action — the pattern any real payments workflow needs. |

**Concepts locked in:**
- State, nodes, edges, and conditional edges as the building blocks of any agent graph
- What `create_react_agent` actually assembles for you
- Checkpointing and thread-scoped memory for stateful, resumable agents
- Human-in-the-loop approval gates via graph interrupts
- Mapping every LangGraph construct back to the raw loop from Week 1

---

## Setup

```bash
# 1. Clone and enter
git clone <your-repo-url> && cd agentic-ai-lab

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install boto3 anthropic langchain-aws langgraph langgraph-checkpoint-sqlite

# 4. Configure credentials and model IDs
cp .env.example .env   # then fill in AWS_REGION and your Bedrock model IDs
```

Confirm your Bedrock inference-profile IDs are current for your region before running:

```bash
aws bedrock list-inference-profiles --region us-east-1 \
  --query 'inferenceProfileSummaries[?contains(inferenceProfileId, `claude`)].inferenceProfileId' \
  --output table
```

Then run any module directly, e.g.:

```bash
python week-01-foundations/03_tool_use.py
```

---

## Roadmap

Weeks 1–2 are the foundation. Upcoming modules build toward a full deployed, evaluated agent:

- [x] **Week 1** — Foundations: raw tool-use loop, provider abstraction
- [x] **Week 2** — LangGraph fundamentals: state, ReAct, streaming, checkpointing, HITL
- [ ] **Week 3** — Multi-agent orchestration: supervisor pattern, streaming across agents
- [ ] **Week 4** — Bedrock AgentCore: deploying a LangGraph agent on managed runtime + memory
- [ ] **Week 5** — Agentic RAG over a knowledge base, with guardrails
- [ ] **Week 6** — Production concerns: evals, observability, cost, security
- [ ] **Capstone** — A domain-relevant agentic assistant demonstrating the full stack

---

## Why this repo exists

Most "I learned agents" repos are a pile of copied tutorials. This one is organized around a single thesis: **you should be able to explain every layer of your agent down to the API call.** The code is written to be read, the notes explain the *why*, and the progression is deliberately bottom-up — primitive first, framework second.
