import json
import logging
from enum import Enum
from typing import Generator

from mistralai.client import Mistral
from pydantic import BaseModel

from pipeline.database import InvoiceDatabase
from pipeline.embeddings import embed_text

logger = logging.getLogger(__name__)

CLASSIFY_MODEL = "mistral-small-latest"
SQL_MODEL = "mistral-small-latest"
CHAT_MODEL = "mistral-large-latest"


class QueryIntent(str, Enum):
    ANALYTICAL = "analytical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


CLASSIFY_PROMPT = """Classify the user's query intent for an invoice database.
Return ONLY one word: analytical, semantic, or hybrid.

- analytical: totals, counts, averages, comparisons, date ranges, specific field lookups (answerable with SQL)
- semantic: finding similar invoices, searching by description/concept, fuzzy matching (needs vector search)
- hybrid: needs both structured data and semantic understanding

Query: {query}
Intent:"""

SQL_SYSTEM_PROMPT = """You are a SQL expert. Given this SQLite schema:
{schema}

Generate a SQL SELECT query to answer the user's question.
Return ONLY the SQL query, no explanation, no markdown fences.
Only SELECT queries are allowed. Use standard SQLite syntax."""


def classify_query(query: str, api_key: str) -> QueryIntent:
    """Classify user query as analytical, semantic, or hybrid."""
    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=CLASSIFY_MODEL,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=10,
        )
        intent_str = response.choices[0].message.content.strip().lower()
        for intent in QueryIntent:
            if intent.value in intent_str:
                return intent
        return QueryIntent.HYBRID
    except Exception as exc:
        logger.warning("Query classification failed: %s", exc)
        return QueryIntent.HYBRID


def build_sql_query(query: str, schema: str, api_key: str) -> str:
    """Generate a SELECT query from natural language."""
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=SQL_MODEL,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=schema)},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    sql = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return sql.strip()


def analytical_retrieval(query: str, db: InvoiceDatabase, api_key: str) -> tuple[str, str]:
    """Text-to-SQL retrieval. Returns (context_string, sql_used)."""
    schema = db.get_schema_description()
    try:
        sql = build_sql_query(query, schema, api_key)
        results = db.execute_sql(sql)
        if results:
            context = f"SQL query: {sql}\n\nResults:\n{json.dumps(results, indent=2, default=str)}"
        else:
            context = f"SQL query: {sql}\n\nNo results found."
        return context, sql
    except Exception as exc:
        logger.warning("Analytical retrieval failed: %s", exc)
        # Fallback: return all invoices as context
        all_inv = db.get_all_invoices()
        context = "All invoices:\n" + json.dumps(all_inv, indent=2, default=str)
        return context, f"(SQL failed: {exc})"


def semantic_retrieval(query: str, db: InvoiceDatabase, api_key: str, top_k: int = 5) -> tuple[str, int]:
    """Vector search retrieval. Returns (context_string, num_results)."""
    try:
        query_embedding = embed_text(query, api_key)
        results = db.search_similar(query_embedding, limit=top_k)
        if results:
            parts = []
            for r in results:
                parts.append(
                    f"- **{r['vendor_name']}** ({r['invoice_number']}) "
                    f"— {r['currency']} {r['total_amount']} "
                    f"[distance: {r.get('distance', 'N/A'):.4f}]\n"
                    f"  Date: {r['invoice_date']}, File: {r['filename']}\n"
                    f"  Text: {(r.get('chunk_text') or '')[:300]}"
                )
            context = "Relevant invoices (by semantic similarity):\n\n" + "\n\n".join(parts)
            return context, len(results)
        else:
            # Fallback if vector search returns nothing
            all_inv = db.get_all_invoices()
            context = "All invoices:\n" + json.dumps(
                [{k: v for k, v in inv.items() if k not in ("raw_markdown", "chunk_text")} for inv in all_inv],
                indent=2, default=str,
            )
            return context, len(all_inv)
    except Exception as exc:
        logger.warning("Semantic retrieval failed: %s", exc)
        all_inv = db.get_all_invoices()
        context = "All invoices:\n" + json.dumps(
            [{k: v for k, v in inv.items() if k not in ("raw_markdown", "chunk_text")} for inv in all_inv],
            indent=2, default=str,
        )
        return context, len(all_inv)


def hybrid_retrieval(query: str, db: InvoiceDatabase, api_key: str) -> tuple[str, str, int]:
    """Run both analytical and semantic retrieval, merge context.
    Returns (context_string, sql_used, num_semantic_results)."""
    analytical_ctx, sql = analytical_retrieval(query, db, api_key)
    semantic_ctx, n = semantic_retrieval(query, db, api_key)
    context = (
        "## Structured Data (SQL)\n" + analytical_ctx +
        "\n\n## Relevant Documents (Semantic Search)\n" + semantic_ctx
    )
    return context, sql, n


def build_rag_context(
    query: str, db: InvoiceDatabase, api_key: str
) -> tuple[str, QueryIntent, dict]:
    """Main RAG entry point: classify -> route -> retrieve.

    Returns (context_string, intent, metadata_dict).
    metadata_dict contains 'sql' and/or 'num_results' for UI display.
    Returns empty context if DB is empty.
    """
    if db.is_empty:
        return "", QueryIntent.HYBRID, {}

    intent = classify_query(query, api_key)
    metadata: dict = {"intent": intent.value}

    if intent == QueryIntent.ANALYTICAL:
        context, sql = analytical_retrieval(query, db, api_key)
        metadata["sql"] = sql
    elif intent == QueryIntent.SEMANTIC:
        context, n = semantic_retrieval(query, db, api_key)
        metadata["num_results"] = n
    else:  # HYBRID
        context, sql, n = hybrid_retrieval(query, db, api_key)
        metadata["sql"] = sql
        metadata["num_results"] = n

    return context, intent, metadata


RAG_SYSTEM_PROMPT = """You are an invoice analysis assistant with access to a database of invoices.

The following context was retrieved to help answer the user's question:

{context}

Answer the user's question based on this data. Be precise with numbers.
If you calculate totals, show your work. Cite specific invoices when possible.
Use markdown formatting."""


def stream_response(
    query: str,
    context: str,
    chat_history: list[dict],
    api_key: str,
) -> Generator[str, None, None]:
    """Stream a response using the RAG context. Yields tokens."""
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": m["role"], "content": m["content"]}
        for m in chat_history
    )

    client = Mistral(api_key=api_key)
    stream = client.chat.stream(
        model=CHAT_MODEL,
        messages=messages,
    )
    for event in stream:
        token = event.data.choices[0].delta.content
        if token:
            yield token
