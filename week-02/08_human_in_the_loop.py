"""Agent that pauses for human approval before executing sensitive tools.
Run: python 08_human_in_loop.py"""
import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from rich import print

load_dotenv()


# ---- Sensitive tool: needs approval ----
@tool
def flag_invoice_for_payment(invoice_id: str, amount: float) -> str:
    """Flag an invoice for payment. This actually triggers a payment run.
    IMPORTANT: requires human approval before executing."""
    approval = interrupt({
        "action": "flag_invoice_for_payment",
        "invoice_id": invoice_id,
        "amount": amount,
        "message": f"Approve payment of ${amount} for {invoice_id}?",
    })
    # execution resumes here after human input
    if approval.get("approved"):
        return f"Payment flagged for {invoice_id} (${amount})."
    return f"Payment for {invoice_id} REJECTED by human. Reason: {approval.get('reason', 'none given')}"


@tool
def get_invoice(invoice_id: str) -> dict:
    """Get invoice details."""
    fake_db = {
        "INV-204": {"vendor": "AcmeCorp", "amount": 4250.00, "status": "unpaid"},
        "INV-206": {"vendor": "Initech", "amount": 12300.00, "status": "unpaid"},
    }
    return fake_db.get(invoice_id, {"error": "not found"})


llm = ChatBedrockConverse(model=os.getenv("BEDROCK_MODEL_ID"),
                          region_name=os.getenv("AWS_REGION"), temperature=0.2)

from langgraph.prebuilt import create_react_agent
checkpointer = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=[get_invoice, flag_invoice_for_payment],
    prompt="You are an invoice assistant. Use tools when needed.",
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "hitl-demo"}}

# First invocation: agent will decide to flag payment, then interrupt
print("[bold cyan]User:[/bold cyan] Look up INV-206 and flag it for payment.\n")
result = agent.invoke(
    {"messages": [HumanMessage("Look up INV-206 and flag it for payment.")]},
    config=config,
)

# Check if we're paused waiting for a human
state = agent.get_state(config)
if state.next:  # non-empty means graph is paused
    print(f"[bold red]PAUSED. Awaiting approval:[/bold red]")
    interrupt_data = state.tasks[0].interrupts[0].value
    print(interrupt_data)
    decision = input("\nApprove? (y/n): ").strip().lower()
    reason = input("Optional reason: ").strip() or "n/a"

    # Resume with the approval
    result = agent.invoke(
        Command(resume={"approved": decision == "y", "reason": reason}),
        config=config,
    )

# Final answer
print(f"\n[bold magenta]claude:[/bold magenta] {result['messages'][-1].content}")