"""Agent with persistent memory across turns via SqliteSaver.
Run: python 07_memory.py"""
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from rich import print
from rich.prompt import Prompt

load_dotenv()


def get_invoice(invoice_id: str) -> dict:
    """Get invoice details."""
    fake_db = {
        "INV-204": {"vendor": "AcmeCorp", "amount": 4250.00, "status": "unpaid", "due": "2026-07-01"},
        "INV-205": {"vendor": "Globex", "amount": 890.50, "status": "paid", "due": "2026-06-15"},
        "INV-206": {"vendor": "Initech", "amount": 12300.00, "status": "unpaid", "due": "2026-06-10"},
    }
    return fake_db.get(invoice_id, {"error": "not found"})

llm = ChatBedrockConverse(model=os.getenv("BEDROCK_MODEL_ID"),
                          region_name=os.getenv("AWS_REGION"), temperature=0.2)

# SqliteSaver persists to a file. Sessions survive across program runs.
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    agent = create_react_agent(
        model=llm,
        tools=[get_invoice],
        prompt="You are an invoice assistant. Remember what the user asked earlier.",
        checkpointer=checkpointer,
    )
    # Thread ID = conversation ID. Same ID = same memory.
    config = {"configurable": {"thread_id": "shantanu-session-1"}}

    print("[bold green]Chat with persistent memory. Type 'exit' to quit.[/bold green]\n")
    while True:
        user_input = Prompt.ask("[bold cyan]you[/bold cyan]")
        if user_input.lower() == "exit":
            break

        result = agent.invoke({"messages": [("user", user_input)]}, config=config)
        last = result["messages"][-1]
        print(f"\n[bold magenta]claude:[/bold magenta] {last.content}\n")