"""LangGraph chatbot: one node that calls the LLM. Add memory next session.
Run: python 03_chat_graph.py"""
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from rich import print
from rich.prompt import Prompt

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
      # Annotated + add_messages is a "reducer": tells LangGraph to APPEND to messages,
      # not replace them. This is the #1 pattern in LangGraph. Memorize it.

llm = ChatBedrockConverse(
    model = os.getenv("BEDROCK_MODEL_ID"),
    region_name=os.getenv("AWS_REGION"),
    temperature=0.3,
    max_tokens=500
)

SYSTEM_PROMPT = (
    "You are teaching a software engineer about agentic AI. "
    "Keep answers under 4 sentences unless asked for depth."
)

def chat_node(state: ChatState) -> dict:
    """Call the LLM with full history + system prompt."""
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT),*state["messages"]])
    return {"messages":[response]}

builder = StateGraph(ChatState)
builder.add_node("chat",chat_node)
builder.add_edge(START,"chat")
builder.add_edge("chat",END)
graph = builder.compile()


if __name__ == "__main__":
    print("[bold green]Chat with Claude via LangGraph. Type 'exit' to quit.[/bold green]\n")
    history: list[BaseMessage] = []

    while True:
        user_input = Prompt.ask("[bold cyan]you[/bold cyan]")
        if user_input.lower() == "exit":
            break
    
        history.append(HumanMessage(user_input))
        result = graph.invoke({"messages":history})
        history = result["messages"]
        
        last: AIMessage = history[-1]
        print(f"\n[bold magenta]claude[/bold magenta]: {last.content}\n")