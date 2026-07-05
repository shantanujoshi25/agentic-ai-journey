"""Verify Bedrock works via LangChain wrapper. Run: python 02_bedrock_llm.py"""
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from rich import print


load_dotenv()

llm = ChatBedrockConverse(
    model=os.getenv("BEDROCK_MODEL_ID"),
    region_name=os.getenv("AWS_REGION"),
    temperature=0.3,
    max_tokens=500,
)

response = llm.invoke("Explain what a LangGraph state machine is in 2 sentences.")
print("[bold green]Response:[/bold green]", response.content)
print("\n[bold cyan]Usage:[/bold cyan]", response.usage_metadata)
