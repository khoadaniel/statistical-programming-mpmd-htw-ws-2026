"""A small MCP server for a made-up campus library, used by workbook 06 of Session 15.

The server offers three tools and one resource:

- search_books: find books by words in the title, the author or the topic.
- check_availability: copies, copies on loan and the next due date of one book.
- late_fee: the fee for books returned late, from the library's rules.
- library://rules: the loan rules as text.

The book list is read from mcp_library_books.json next to this file. The file is opened
read-only, and no tool writes anything, so the server cannot change the data.

Run it over stdio (the default, started by an MCP client):

    python mcp_server_library.py

Run it over Streamable HTTP on this computer only:

    python mcp_server_library.py --http --port 8765

A stdio server must never print to standard output: that stream carries the protocol
messages. Log to standard error instead.
"""

import argparse
import json
from pathlib import Path

from fastmcp import FastMCP

DATA_FILE = Path(__file__).with_name("mcp_library_books.json")
FEE_PER_BOOK_PER_DAY_EUR = 0.20
FEE_CAP_PER_BOOK_EUR = 10.00

RULES = f"""Loan rules of the Spree Campus Library (made up for this workbook)

- A loan lasts 28 days. You can renew a loan twice, if nobody has reserved the book.
- A late book costs {FEE_PER_BOOK_PER_DAY_EUR:.2f} EUR per day, at most {FEE_CAP_PER_BOOK_EUR:.2f} EUR per book.
- Students can borrow up to 15 books at the same time.
"""

mcp = FastMCP("spree-campus-library")


def load_books() -> list[dict]:
    """Read the book list from the JSON file, opened read-only."""
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)["books"]


@mcp.tool
def search_books(query: str, max_results: int = 5) -> list[dict]:
    """Search the library catalogue. Matches every word of the query against title, author and topic.

    Use it first to find a book's id. Returns at most max_results books with id, title, author, year and topic.
    An empty list means that no book matches.
    """
    words = query.lower().split()
    found = []
    for book in load_books():
        text = f"{book['title']} {book['author']} {book['topic']}".lower()
        if all(word in text for word in words):
            found.append({key: book[key] for key in ("id", "title", "author", "year", "topic")})
    return found[:max_results]


@mcp.tool
def check_availability(book_id: str) -> dict:
    """Check how many copies of one book are free now, and when the next copy is due back.

    book_id is the id from search_books, for example "B01".
    """
    for book in load_books():
        if book["id"] == book_id.upper():
            free = book["copies"] - book["on_loan"]
            return {
                "id": book["id"],
                "title": book["title"],
                "copies": book["copies"],
                "free_now": free,
                "next_due": book["next_due"] if free == 0 else None,
            }
    return {"error": f"No book with id {book_id!r}. Call search_books to find the right id."}


@mcp.tool
def late_fee(days_late: int, books: int = 1) -> dict:
    """Compute the late fee in EUR for books returned days_late days after the due date.

    Applies the library's rules: a fee per book per day, with a cap per book.
    """
    if days_late < 0 or books < 1:
        return {"error": "days_late must be 0 or more and books must be 1 or more."}
    per_book = min(days_late * FEE_PER_BOOK_PER_DAY_EUR, FEE_CAP_PER_BOOK_EUR)
    return {
        "days_late": days_late,
        "books": books,
        "fee_per_book_eur": round(per_book, 2),
        "total_eur": round(per_book * books, 2),
        "capped": per_book == FEE_CAP_PER_BOOK_EUR,
    }


@mcp.resource("library://rules")
def loan_rules() -> str:
    """The loan rules of the library as plain text."""
    return RULES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spree Campus Library MCP server")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.http:
        # 127.0.0.1 means: only programs on this computer can connect.
        mcp.run(transport="http", host="127.0.0.1", port=arguments.port, show_banner=False)
    else:
        mcp.run(show_banner=False)
