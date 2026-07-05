"""Rebuild the invoice agent from Week 1's 03_tool_use.py as a LangGraph graph.
Run: python 04_tools_graph.py"""
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich import print

load_dotenv()

@tool
def get_invoice(invoice_id:str)->dict:
    """Get full details for a single invoice by its ID."""
    fake_db = {
        "INV-204": {"vendor": "AcmeCorp", "amount": 4250.00, "status": "unpaid", "due": "2026-07-01"},
        "INV-205": {"vendor": "Globex", "amount": 890.50, "status": "paid", "due": "2026-06-15"},
        "INV-206": {"vendor": "Initech", "amount": 12300.00, "status": "unpaid", "due": "2026-06-10"},
    }
    return fake_db.get(invoice_id, {"error": f"Invoice {invoice_id} not found"})


@tool
def list_unpaid_invoices() -> list:
    """List all unpaid invoices in the system."""
    return [
        {"id": "INV-204", "vendor": "AcmeCorp", "amount": 4250.00, "due": "2026-07-01"},
        {"id": "INV-206", "vendor": "Initech", "amount": 12300.00, "due": "2026-06-10"},
    ]

TOOLS = [get_invoice,list_unpaid_invoices]

llm = ChatBedrockConverse(
    model=os.getenv("BEDROCK_MODEL_ID"),
    region_name=os.getenv("AWS_REGION"),
    temperature=0.3,
    max_tokens=800
)

llm_with_tools = llm.bind_tools(TOOLS)

class AgentState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]
    
SYSTEM_PROMPT = "You are an invoice assistant. Use the tools to look up real data."

def agent_node(state: AgentState) -> dict:
    """The LLM node: decides whether to call tools or answer."""
    response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(TOOLS)


# ---- Conditional edge: after the agent responds, do we need tools? ----
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    # If the last AI message has tool_calls, run tools; else we're done.
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


# ---- Build the graph ----
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue,
                              {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")   # after tools run, loop back to agent
graph = builder.compile()

def run(question: str):
    print(f"\n[bold cyan]User:[/bold cyan] {question}\n")
    for event in graph.stream({"messages": [HumanMessage(question)]},
                              stream_mode="updates"):
        for node_name, update in event.items():
            for msg in update["messages"]:
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"[bold magenta]claude:[/bold magenta] {msg.content}")
                    for tc in msg.tool_calls:
                        print(f"[yellow]-> tool call:[/yellow] {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"[green]<- tool result:[/green] {msg.content[:200]}")

if __name__ == "__main__":
    print("[bold]Graph:[/bold]")
    print(graph.get_graph().draw_ascii())

    run("Which invoices are unpaid and which is most urgent?")
    run("What's the status of INV-205?")