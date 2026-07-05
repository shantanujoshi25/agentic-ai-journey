"""The same agent as 04, using create_react_agent. This is what most people ship.
Run: python 05_react_prebuilt.py"""
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich import print

load_dotenv()

@tool
def get_invoice(invoice_id: str) -> dict:
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
llm = ChatBedrockConverse(
    model=os.getenv("BEDROCK_MODEL_ID"),
    region_name=os.getenv("AWS_REGION"),
    temperature=0.2,
)
agent =create_react_agent(
    model = llm,
    tools = [get_invoice,list_unpaid_invoices],
    prompt = "You are an invoice assistant. Use the tools to look up real data.",
)

if __name__ == "__main__":
    print("[bold]Graph:[/bold]")
    print(agent.get_graph().draw_ascii())

    for question in ["Which invoices are unpaid and which is most urgent?",
                     "What's the status of INV-205?"]:
        print(f"\n[bold cyan]User:[/bold cyan] {question}\n")
        result = agent.invoke({"messages": [("user", question)]})
        print(result)
        print(f"[bold magenta]claude:[/bold magenta] {result['messages'][-1].content}\n")
        # stream_mode="updates" — see each node's contribution
        
