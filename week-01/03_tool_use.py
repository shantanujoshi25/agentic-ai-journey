"""Raw tool use loop with Bedrock Converse. Run: python 03_tool_use.py"""
import os
import json
import boto3
from dotenv import load_dotenv
from rich import print

load_dotenv()
client = boto3.Session(region_name=os.getenv("AWS_REGION")).client("bedrock-runtime")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

def get_invoice(invoice_id:str) -> dict:
    """Mock invoice lookup"""
    fake_db = {
        "INV-204": {"vendor": "AcmeCorp", "amount": 4250.00, "status": "unpaid", "due": "2026-07-01"},
        "INV-205": {"vendor": "Globex", "amount": 890.50, "status": "paid", "due": "2026-06-15"},
        "INV-206": {"vendor": "Initech", "amount": 12300.00, "status": "unpaid", "due": "2026-06-10"},
    }
    return fake_db.get(invoice_id, {"error": f"Invoice {invoice_id} not found"})

def list_unpaid_invoices() -> list:
    """Mock list of unpaid invoices."""
    return[
        {"id": "INV-204", "vendor": "AcmeCorp", "amount": 4250.00, "due": "2026-07-01"},
        {"id": "INV-206", "vendor": "Initech", "amount": 12300.00, "due": "2026-06-10"},
    ]
    
TOOLS = {
    "get_invoice": get_invoice,
    "list_unpaid_invoices": list_unpaid_invoices,
}


# --- 2. Describe tools to the model (Bedrock format) ---
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_invoice",
                "description": "Get full details for a single invoice by its ID.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "invoice_id": {"type": "string", "description": "Invoice ID like 'INV-204'"}
                        },
                        "required": ["invoice_id"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "list_unpaid_invoices",
                "description": "List all unpaid invoices in the system.",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        },
    ]
}

def run(user_question:str, max_turns: int = 5):
    messages = [{"role":"user","content":[{"text":user_question}]}]
    print(f"[bold cyan]User:[/bold cyan] {user_question}\n")

    for turn in range(max_turns):
        response = client.converse(
            modelId = MODEL_ID,
            messages = messages,
            toolConfig = TOOL_CONFIG,
            inferenceConfig = {"maxTokens":800, "temperature": 0.2},
        )
        
        assistant_msg = response["output"]["message"]
        messages.append(assistant_msg)
        stop_Reason = response['stopReason']
        print(response)
        for block in assistant_msg["content"]:
            if "text" in block:
                print(f"[bold magenta]Claude:[/bold magenta] {block['text']}")
            if "toolUse" in block:
                tu = block["toolUse"]
                print(f"[bold yellow]-> tool call:[/bold yellow] {tu['name']}({tu['input']})")

        if stop_Reason == "end_turn":
            print("\n[green]Done.[/green]")
            return

        if stop_Reason == "tool_use":
            # Execute each requested tool, collect results
            tool_results = []
            for block in assistant_msg["content"]:
                if "toolUse" not in block:
                    continue
                tu = block["toolUse"]
                fn = TOOLS[tu["name"]]
                result = fn(**tu["input"])
                print(f"[bold green]<- tool result:[/bold green] {result}")
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"json": result if isinstance(result, dict) else {"items": result}}],
                    }
                })
                messages.append({"role": "user", "content": tool_results})
    print("\n[red]Hit max turns without finishing.[/red]")
    
if __name__ == "__main__":
    run("Which invoices are unpaid and which is most urgent? Give me a 1-sentence summary.",1)
    print("\n---\n")
    run("What's the status of INV-205?")