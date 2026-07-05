"""First Anthropic API call. Run: python 01_hello_anthropic.py"""
import os
import boto3
from dotenv import load_dotenv
from rich import print

load_dotenv()

REGION = os.getenv("AWS_REGION","us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

session = boto3.Session(region_name = REGION)
client = session.client("bedrock-runtime")

response = client.converse(
    modelId = MODEL_ID,
    messages = [
        {
            "role":"user",
            "content":[{"text":"Explain Agentic AI in 3 sentences for a software engineer"}]
        }
    ],
    inferenceConfig = {
        "maxTokens" : 500,
        "temperature":0.3
    }
)


# client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# response = client.messages.create(
#     model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
#     max_tokens=500,
#     messages=[
#         {"role": "user", "content": "Explain agentic AI in 3 sentences for a software engineer."}
#     ],
# )


print("\n[bold green]Response:[/bold green]")
print(response["output"]["message"]["content"][0]["text"])

print("\n[bold cyan]Usage:[/bold cyan]")
print(response["usage"])

print("\n[bold cyan]Stop reason:[/bold cyan]", response["stopReason"])