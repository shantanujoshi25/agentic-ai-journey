"""Multi-turn chat with Claude on Bedrock. Run: python 02_chat.py"""
import os
import boto3
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

load_dotenv()

REGION = os.getenv("AWS_REGION","us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

client = boto3.Session(region_name=REGION).client("bedrock-runtime")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant teaching a software engineer about agentic AI. "
    "Keep answers under 4 sentences unless asked for depth. Use concrete examples."
)
messages = []
total_in = 0
total_out = 0

print("[bold green]Chat with Claude. Type 'exit' to quit, 'tokens' for usage.[/bold green]\n")

while True:
    user_input = Prompt.ask("[bold cyan]you[/bold cyan]")
    if user_input.lower() == "exit":
        break
    if user_input.lower() == "tokens":
        print(f"[yellow]Input tokens so far: {total_in}, output: {total_out}[/yellow]")
        continue

    messages.append({"role":"user","content":[{"text":user_input}]})
    response = client.converse(
        modelId = MODEL_ID,
        messages = messages,
        system = [{"text": SYSTEM_PROMPT}],
        inferenceConfig = {"maxTokens":600,"temperature":0.3},
    )    
    
    
    assistant_message = response["output"]["message"]
    messages.append(assistant_message)
    
    total_in += response["usage"]["inputTokens"]
    total_out += response["usage"]["outputTokens"]

    print(f"\n[bold magenta]claude[/bold magenta]: {assistant_message['content'][0]['text']}\n")