import logging
import os
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from pipeline.schemas import ProcessedInvoice

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "invoices.db"


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


class InvoiceDatabase:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        self._vec_available = False

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        # Test vec extension availability and init tables
        conn = self._connect()
        try:
            self._init_tables(conn)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            self._vec_available = True
        except Exception as exc:
            logger.warning("sqlite-vec unavailable: %s", exc)
            self._vec_available = False
        return conn

    def _init_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                vendor_name TEXT NOT NULL,
                vendor_address TEXT,
                invoice_number TEXT NOT NULL,
                invoice_date TEXT NOT NULL,
                due_date TEXT,
                currency TEXT NOT NULL DEFAULT 'USD',
                subtotal REAL NOT NULL,
                tax_amount REAL,
                total_amount REAL NOT NULL,
                payment_terms TEXT,
                extraction_method TEXT NOT NULL,
                raw_markdown TEXT,
                chunk_text TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS line_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id INTEGER NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
                description TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit_price REAL NOT NULL,
                total REAL NOT NULL
            );
        """)
        if self._vec_available:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS invoice_embeddings USING vec0(
                    invoice_id INTEGER PRIMARY KEY,
                    embedding float[1024]
                )
            """)
        conn.commit()

    def save_invoice(
        self,
        processed: ProcessedInvoice,
        chunk_text: str = "",
        embedding: list[float] | None = None,
    ) -> int:
        """Insert or replace invoice + line items + embedding. Returns invoice id."""
        conn = self._connect()
        try:
            data = processed.data
            # Delete existing record if re-saving same filename
            existing = conn.execute(
                "SELECT id FROM invoices WHERE filename = ?", (processed.filename,)
            ).fetchone()
            if existing:
                self._delete_invoice_internal(conn, existing["id"])

            cursor = conn.execute(
                """INSERT INTO invoices
                   (filename, vendor_name, vendor_address, invoice_number,
                    invoice_date, due_date, currency, subtotal, tax_amount,
                    total_amount, payment_terms, extraction_method,
                    raw_markdown, chunk_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    processed.filename,
                    data.vendor_name,
                    data.vendor_address,
                    data.invoice_number,
                    data.invoice_date,
                    data.due_date,
                    data.currency,
                    data.subtotal,
                    data.tax_amount,
                    data.total_amount,
                    data.payment_terms,
                    processed.extraction_method,
                    processed.raw_markdown,
                    chunk_text,
                ),
            )
            invoice_id = cursor.lastrowid

            for item in data.line_items:
                conn.execute(
                    """INSERT INTO line_items
                       (invoice_id, description, quantity, unit_price, total)
                       VALUES (?, ?, ?, ?, ?)""",
                    (invoice_id, item.description, item.quantity, item.unit_price, item.total),
                )

            if embedding and self._vec_available:
                try:
                    conn.execute(
                        "INSERT INTO invoice_embeddings (invoice_id, embedding) VALUES (?, ?)",
                        (invoice_id, _serialize_f32(embedding)),
                    )
                except Exception as exc:
                    logger.warning("Failed to store embedding: %s", exc)

            conn.commit()
            return invoice_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_all_invoices(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT i.*, COUNT(li.id) as line_items_count
                   FROM invoices i
                   LEFT JOIN line_items li ON li.invoice_id = i.id
                   GROUP BY i.id
                   ORDER BY i.created_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_invoice_by_filename(self, filename: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM invoices WHERE filename = ?", (filename,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_line_items(self, invoice_id: int) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM line_items WHERE invoice_id = ?", (invoice_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_invoice(self, invoice_id: int) -> None:
        conn = self._connect()
        try:
            self._delete_invoice_internal(conn, invoice_id)
            conn.commit()
        finally:
            conn.close()

    def _delete_invoice_internal(self, conn: sqlite3.Connection, invoice_id: int) -> None:
        """Delete invoice and related data within an existing connection/transaction."""
        if self._vec_available:
            try:
                conn.execute(
                    "DELETE FROM invoice_embeddings WHERE invoice_id = ?", (invoice_id,)
                )
            except Exception as exc:
                logger.warning("Failed to delete embedding: %s", exc)
        conn.execute("DELETE FROM line_items WHERE invoice_id = ?", (invoice_id,))
        conn.execute("DELETE FROM invoices WHERE id = ?", (invoice_id,))

    def search_similar(self, query_embedding: list[float], limit: int = 5) -> list[dict]:
        """Vector similarity search via sqlite-vec."""
        if not self._vec_available:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT i.*, e.distance
                   FROM invoice_embeddings e
                   JOIN invoices i ON i.id = e.invoice_id
                   WHERE e.embedding MATCH ?
                     AND k = ?
                   ORDER BY e.distance""",
                (_serialize_f32(query_embedding), limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []
        finally:
            conn.close()

    def execute_sql(self, query: str) -> list[dict]:
        """Execute a read-only SQL query. Only SELECT statements allowed."""
        stripped = query.strip()
        if not stripped.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        conn = self._connect()
        try:
            rows = conn.execute(stripped).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT
                     COUNT(*) as count,
                     COALESCE(SUM(total_amount), 0) as total_spend,
                     COUNT(DISTINCT vendor_name) as vendor_count,
                     COALESCE(AVG(total_amount), 0) as avg_amount
                   FROM invoices"""
            ).fetchone()
            return dict(row)
        finally:
            conn.close()

    def get_schema_description(self) -> str:
        """Return DDL description for the LLM's text-to-SQL prompt."""
        return """
Tables:
- invoices(id, filename, vendor_name, vendor_address, invoice_number,
  invoice_date, due_date, currency, subtotal, tax_amount, total_amount,
  payment_terms, extraction_method, raw_markdown, created_at)
- line_items(id, invoice_id, description, quantity, unit_price, total)

Relationships: line_items.invoice_id -> invoices.id
"""

    @property
    def is_empty(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) as c FROM invoices").fetchone()
            return row["c"] == 0
        finally:
            conn.close()

    def clear_all(self) -> None:
        """Delete all data from all tables."""
        conn = self._connect()
        try:
            if self._vec_available:
                try:
                    # vec0 tables need row-by-row delete
                    conn.execute(
                        "DELETE FROM invoice_embeddings WHERE invoice_id IN (SELECT id FROM invoices)"
                    )
                except Exception as exc:
                    logger.warning("Failed to clear embeddings: %s", exc)
            conn.execute("DELETE FROM line_items")
            conn.execute("DELETE FROM invoices")
            conn.commit()
        finally:
            conn.close()
