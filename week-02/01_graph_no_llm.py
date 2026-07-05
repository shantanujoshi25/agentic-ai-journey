"""A LangGraph graph with zero LLMs. Just state, nodes, edges.
Purpose: internalize what LangGraph actually is.
Run: python 01_graph_no_llm.py"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rich import print


# ---- 1. State: what flows through the graph ----
class OrderState(TypedDict):
    order_id: str
    items: list[str]
    total: float
    status: str          # "new" -> "priced" -> "validated" -> "confirmed"
    is_valid: bool
    reason: str          # explanation if invalid


# ---- 2. Nodes: pure functions that read + write state ----
def price_order(state: OrderState) -> dict:
    """Compute total based on items."""
    print(f"[yellow]node:[/yellow] price_order")
    prices = {"pen": 2.0, "notebook": 5.0, "laptop": 900.0, "chair": 150.0}
    total = sum(prices.get(item, 0) for item in state["items"])
    return {"total": total, "status": "priced"}


def validate_order(state: OrderState) -> dict:
    """Reject if any item has no price, or total > 1000."""
    print(f"[yellow]node:[/yellow] validate_order")
    unknown_items = [i for i in state["items"] if i not in {"pen", "notebook", "laptop", "chair"}]
    if unknown_items:
        return {"is_valid": False, "status": "invalid",
                "reason": f"Unknown items: {unknown_items}"}
    if state["total"] > 1000:
        return {"is_valid": False, "status": "invalid",
                "reason": f"Total ${state['total']} exceeds $1000 limit"}
    return {"is_valid": True, "status": "validated", "reason": ""}


def confirm_order(state: OrderState) -> dict:
    print(f"[yellow]node:[/yellow] confirm_order")
    return {"status": "confirmed"}


def reject_order(state: OrderState) -> dict:
    print(f"[yellow]node:[/yellow] reject_order (reason: {state['reason']})")
    return {"status": "rejected"}


# ---- 3. Conditional edge: decides where to go after validate ----
def route_after_validate(state: OrderState) -> str:
    return "confirm" if state["is_valid"] else "reject"


# ---- 4. Build the graph ----
builder = StateGraph(OrderState)

builder.add_node("price", price_order)
builder.add_node("validate", validate_order)
builder.add_node("confirm", confirm_order)
builder.add_node("reject", reject_order)

builder.add_edge(START, "price")
builder.add_edge("price", "validate")
builder.add_conditional_edges("validate", route_after_validate,
                              {"confirm": "confirm", "reject": "reject"})
builder.add_edge("confirm", END)
builder.add_edge("reject", END)

graph = builder.compile()


# ---- 5. Run it three times with different inputs ----
if __name__ == "__main__":
    print("\n[bold cyan]--- Case 1: valid small order ---[/bold cyan]")
    result = graph.invoke({
        "order_id": "O-001", "items": ["pen", "notebook", "chair"],
        "total": 0.0, "status": "new", "is_valid": False, "reason": "",
    })
    print(result)

    print("\n[bold cyan]--- Case 2: too expensive ---[/bold cyan]")
    result = graph.invoke({
        "order_id": "O-002", "items": ["laptop", "chair"],
        "total": 0.0, "status": "new", "is_valid": False, "reason": "",
    })
    print(result)

    print("\n[bold cyan]--- Case 3: unknown item ---[/bold cyan]")
    result = graph.invoke({
        "order_id": "O-003", "items": ["pen", "hamster"],
        "total": 0.0, "status": "new", "is_valid": False, "reason": "",
    })
    print(result)
    print("\n[bold cyan]Graph structure:[/bold cyan]")
    print(graph.get_graph().draw_ascii())