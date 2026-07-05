# client.py — a host/client that lets Claude discover + call the MCP tools
import asyncio
from anthropic import Anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv


load_dotenv()

anthropic = anthropic = Anthropic()
MCP_URL = "http://127.0.0.1:8000/mcp"
MODEL   = "claude-sonnet-5"        # swap for any current model you have API access to


def to_anthropic_tools(mcp_tools):
    # MCP hands you JSON Schema already (t.inputSchema) — just remap the key name.
    return [{"name": t.name, "description": t.description or "",
             "input_schema": t.inputSchema} for t in mcp_tools]
    
    
    
    
async def run(question: str):
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()                    # MCP handshake
            listed = await session.list_tools()           # (1) DISCOVERY
            tools = to_anthropic_tools(listed.tools)
            print("Discovered:", [t["name"] for t in tools])

            messages = [{"role": "user", "content": question}]
            while True:                                    # the agent loop
                resp = anthropic.messages.create(
                    model=MODEL, max_tokens=1024, tools=tools, messages=messages)
                messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use":         # Claude gave a final answer
                    print("\nClaude:", "".join(b.text for b in resp.content if b.type == "text"))
                    return

                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    print(f"-> {block.name}({block.input})")
                    out = await session.call_tool(block.name, block.input)   # (2) INVOCATION
                    text = "\n".join(getattr(c, "text", "") for c in out.content) or "(no output)"
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id, "content": text})
                messages.append({"role": "user", "content": results})        # feed results back

if __name__ == "__main__":
    asyncio.run(run(
        "Which unpaid invoices are over $10,000? Flag the largest one for review as a possible duplicate."))