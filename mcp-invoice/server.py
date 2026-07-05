# server.py — a minimal MCP server exposed over streamable HTTP
import sqlite3
from mcp.server.fastmcp import FastMCP


# host/port define the endpoint: http://127.0.0.1:8000/mcp
mcp = FastMCP("invoice-ops", host="127.0.0.1", port=8000)

db = sqlite3.connect(":memory:", check_same_thread=False)  # check_same_thread=False: demo only
db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE invoices (
    id TEXT PRIMARY KEY, vendor TEXT, amount REAL, currency TEXT,
    status TEXT, due_date TEXT, flagged INTEGER DEFAULT 0, flag_reason TEXT
);
INSERT INTO invoices (id, vendor, amount, currency, status, due_date) VALUES
    ('INV-1001','Acme Steel',       12500.00,'USD','unpaid','2026-07-10'),
    ('INV-1002','Globex Logistics',  3200.50,'USD','paid',  '2026-06-15'),
    ('INV-1003','Initech Cloud',      899.00,'USD','unpaid','2026-07-01'),
    ('INV-1004','Umbrella Supplies', 54000.00,'USD','unpaid','2026-07-20');
""")
db.commit()


@mcp.tool()
def get_invoice(invoice_id: str) -> dict:
    """Look up a single invoice by ID (e.g. 'INV-1001')."""
    row = db.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
    return dict(row) if row else {"error": f"No invoice {invoice_id}"}

@mcp.tool()
def list_unpaid_invoices(min_amount: float = 0.0) -> list[dict]:
    """List unpaid invoices at or above min_amount, sorted by due date."""
    rows = db.execute(
        "SELECT * FROM invoices WHERE status='unpaid' AND amount >= ? ORDER BY due_date",
        (min_amount,),
    ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def flag_invoice_for_review(invoice_id: str, reason: str) -> dict:
    """Flag an invoice for human review with a reason. This mutates state (a side effect)."""
    cur = db.execute(
        "UPDATE invoices SET flagged=1, flag_reason=? WHERE id=?", (reason, invoice_id)
    )
    db.commit()
    return ({"invoice_id": invoice_id, "flagged": True, "reason": reason}
            if cur.rowcount else {"error": f"No invoice {invoice_id}"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # serves http://127.0.0.1:8000/mcp
