#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Elephant Gun CLI
# - Local-first hybrid SQL + semantic search for PostgreSQL
# - Keep this file thin. Heavy logic for `scan` lives in elephant_gun/eg_scan.py
# - DB connection helper is shared via elephant_gun/db.py

import re
import sys
import argparse
from typing import List, Optional
from datetime import datetime, timedelta, timezone

import yaml
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# Package-local imports
from .db import conn                   # shared DB connection helper
from .eg_scan import scan_schema       # separated `scan` implementation


# ---------- Config models ----------
class Target(BaseModel):
    table: str
    key: str
    time_column: Optional[str] = None
    text_template: str
    filters: List[str] = []  # optional static SQL filters


class Config(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    targets: List[Target]


# ---------- Config I/O ----------
def load_cfg(path: str = "elephant_gun.yaml") -> Config:
    """Load CLI configuration (model + embedding targets)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# ---------- Extension / schema helpers ----------
def ensure_ext():
    """Ensure pgvector extension is available."""
    with conn() as con:
        con.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("vector extension ensured.")


def ensure_target_schema(t: Target, embed_dim: int):
    """Ensure embedding column and helpful indexes exist for a target table."""
    with conn() as con:
        con.execute(
            f"ALTER TABLE {t.table} "
            f"ADD COLUMN IF NOT EXISTS embedding VECTOR({embed_dim});"
        )
        if t.time_column:
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{t.table}_{t.time_column} "
                f"ON {t.table} ({t.time_column} DESC);"
            )
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{t.table}_{t.key} "
            f"ON {t.table} ({t.key});"
        )


# ---------- Embedding ----------
_model = None  # lazy-loaded global to avoid re-loading the model


def get_model(cfg: Config):
    """Load (or reuse) the embedding model specified in config."""
    global _model
    if _model is None:
        _model = SentenceTransformer(cfg.model)
    return _model


def render_text(row: dict, template: str) -> str:
    """Render a Jinja-lite template of '{{col}}' placeholders using row dict."""
    out = template
    for k, v in row.items():
        out = out.replace("{{" + k + "}}", "" if v is None else str(v))
    # Collapse excessive whitespace
    return re.sub(r"\s+", " ", out).strip()


def iter_batch_rows_to_embed(t: Target, batch: int = 500):
    """Yield rows (dicts) needing embeddings for the given target table."""
    # Collect the columns referenced by the template plus the key
    cols_needed = {t.key}
    cols_needed |= set(re.findall(r"{{\s*([a-zA-Z0-9_]+)\s*}}", t.text_template))
    select_cols = ", ".join(sorted(cols_needed))

    with conn() as con:
        # Stream by repeatedly fetching LIMIT chunks of rows with NULL embeddings
        while True:
            rows = con.execute(
                f"SELECT {select_cols} FROM {t.table} "
                f"WHERE embedding IS NULL LIMIT %s",
                (batch,),
            ).fetchall()
            if not rows:
                break

            # Fetch column names for dict conversion
            names = [
                d.name for d in con.execute(
                    f"SELECT {select_cols} FROM {t.table} LIMIT 0"
                ).description
            ]
            dicts = [dict(zip(names, r)) for r in rows]
            yield dicts


def backfill_embeddings(cfg: Config, table: Optional[str] = None, batch: int = 500):
    """Compute embeddings for rows lacking vectors (optionally for one table)."""
    model = get_model(cfg)
    targets = [t for t in cfg.targets if (table is None or t.table == table)]

    for t in targets:
        ensure_target_schema(t, cfg.embed_dim)
        total = 0
        while True:
            chunk = next(iter_batch_rows_to_embed(t, batch), [])
            if not chunk:
                break

            texts = [render_text(r, t.text_template) for r in chunk]
            vecs = model.encode(texts, normalize_embeddings=True)

            with conn() as con:
                with con.cursor() as cur:
                    for r, v in zip(chunk, vecs):
                        vec_list = ",".join(f"{x:.7f}" for x in v.tolist())
                        cur.execute(
                            f"UPDATE {t.table} SET embedding='[{vec_list}]'::vector "
                            f"WHERE {t.key}=%s",
                            (r[t.key],),
                        )
            total += len(chunk)
            print(f"[{t.table}] embedded +{len(chunk)} (total {total})")

        print(f"[{t.table}] embedding done: {total}")


def embed_query(cfg: Config, text: str):
    """Encode a single query string into a normalized vector."""
    model = get_model(cfg)
    return model.encode(text, normalize_embeddings=True)

def parse_time_window_from_text(text: str) -> tuple[Optional[int], str]:
    """
    Parse light-weight time expressions from natural language and return:
      (days, cleaned_text)
    Supported:
      - "last N day(s)|week(s)|month(s)"  (weeks=7N, months≈30N)
      - "past N days|weeks|months"
      - "yesterday" (days=1)
      - "today" (days=1)  # practical rolling day
      - "last week" (days=7)
      - "last month" (days=30)
      - "since YYYY-MM-DD"  -> days = (today - date).days
      - "YYYY-MM-DD..YYYY-MM-DD" -> days = (end - start).days + 1
    It also removes the matched phrase(s) from the query text.
    """
    original = text
    s = text.lower()

    # canonicalize whitespace
    s = re.sub(r"\s+", " ", s)

    # date range: 2025-09-01..2025-09-30
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\s*\.\.\s*(\d{4}-\d{2}-\d{2})\b", s)
    if m:
        try:
            d1 = datetime.fromisoformat(m.group(1))
            d2 = datetime.fromisoformat(m.group(2))
            if d2 >= d1:
                days = (d2 - d1).days + 1
                cleaned = (s[:m.start()] + s[m.end():]).strip()
                return days, cleaned
        except ValueError:
            pass  # fallthrough

    # since YYYY-MM-DD
    m = re.search(r"\bsince\s+(\d{4}-\d{2}-\d{2})\b", s)
    if m:
        try:
            d1 = datetime.fromisoformat(m.group(1))
            today = datetime.now(timezone.utc).date()
            days = max(1, (today - d1.date()).days + 1)
            cleaned = (s[:m.start()] + s[m.end():]).strip()
            return days, cleaned
        except ValueError:
            pass

    if re.search(r"\bthis week\b", s):
        today = datetime.now(timezone.utc).date()
        # Monday = 0
        start = today - timedelta(days=today.weekday())
        days = (today - start).days + 1
        cleaned = re.sub(r"\bthis week\b", "", s).strip()
        return days, cleaned

    if re.search(r"\bthis month\b", s):
        today = datetime.now(timezone.utc).date()
        start = today.replace(day=1)
        days = (today - start).days + 1
        cleaned = re.sub(r"\bthis month\b", "", s).strip()
        return days, cleaned
    
    # yesterday / today
    if re.search(r"\byesterday\b", s):
        cleaned = re.sub(r"\byesterday\b", "", s).strip()
        return 1, cleaned
    if re.search(r"\btoday\b", s):
        cleaned = re.sub(r"\btoday\b", "", s).strip()
        return 1, cleaned

    # last week / last month
    if re.search(r"\blast week\b", s):
        cleaned = re.sub(r"\blast week\b", "", s).strip()
        return 7, cleaned
    if re.search(r"\blast month\b", s):
        cleaned = re.sub(r"\blast month\b", "", s).strip()
        return 30, cleaned

    # last N units / past N units
    m = re.search(r"\b(last|past)\s+(\d+)\s*(day|days|week|weeks|month|months)\b", s)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        if "week" in unit:
            days = n * 7
        elif "month" in unit:
            days = n * 30
        else:
            days = n
        cleaned = (s[:m.start()] + s[m.end():]).strip()
        return max(1, days), cleaned

    # no time hints found
    return None, original


# ---------- Query ----------
def sql_filters(t: Target, days: Optional[int]) -> str:
    """Compose WHERE fragment from static filters and optional time window."""
    parts = []
    # Static filters from config
    for f in t.filters:
        parts.append(f"({f})")
    # Optional rolling time window
    if days and t.time_column:
        parts.append(f"{t.time_column} >= now() - interval '{int(days)} days'")
    return " AND ".join(parts) if parts else "TRUE"


def hybrid_query(
    cfg: Config,
    table: str,
    q: str,
    days: Optional[int],
    limit: int,
    dry_run: bool,
    min_sim: Optional[float],   # <-- NEW: filter by sim01 threshold (0..1)
):
    if days is None:
        parsed_days, cleaned = parse_time_window_from_text(q)
        if parsed_days is not None:
            days = parsed_days
            q = cleaned or q  # if fully removed, keep original to avoid empty text

    """Run a hybrid (vector-first) query on a configured target table."""
    qvec = embed_query(cfg, q)
    vec_list = ",".join(f"{x:.7f}" for x in qvec.tolist())

    matches = [x for x in cfg.targets if x.table == table]
    if not matches:
        print(f"target table not found in elephant_gun.yaml: {table}", file=sys.stderr)
        sys.exit(1)
    t = matches[0]

    where = sql_filters(t, days)
    # Build sim expression once to reuse in WHERE and SELECT
    sim_expr = f"(1 - 0.5 * (embedding <=> '[{vec_list}]'::vector))"
    where_extra = f" AND {sim_expr} >= {min_sim:.3f}" if min_sim is not None else ""

    # Minimal preview logic for demo; users can customize per table
    preview = (
        "left(coalesce(title,'')||' '||coalesce(body,''), 120) as preview"
        if t.table == "tickets"
        else "NULL as preview"
    )

    sql = f"""
    SELECT {t.key} AS id, {preview},
           1 - (embedding <=> '[{vec_list}]'::vector) AS cosine_sim,   -- [-1, 1]
           {sim_expr} AS sim01                                        -- [0, 1] for display
    FROM {t.table}
    WHERE {where} AND embedding IS NOT NULL{where_extra}
    ORDER BY embedding <=> '[{vec_list}]'::vector
    LIMIT {int(limit)};
    """.strip()

    if dry_run:
        print(sql)
        return

    with conn() as con:
        rows = con.execute(sql).fetchall()

    # Pretty print
    shown = 0
    for r in rows:
        rid, prev, sim, sim01 = r
        print(f"[{rid}] sim01={sim01:.3f}  {prev}")
        shown += 1
    if shown == 0 and min_sim is not None:
        # Friendly hint when threshold is too strict
        print(f"(no results ≥ min-sim {min_sim:.2f}; try lowering --min-sim or widening --days)")


def init_targets(cfg: Config):
    """Ensure required schema objects for all targets (embedding column + indexes)."""
    for t in cfg.targets:
        ensure_target_schema(t, cfg.embed_dim)
    print("init done. (embedding column/index ensured)")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        prog="elephant-gun",
        description="Hybrid SQL + semantic search CLI for existing Postgres.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Lightweight subcommands first
    sub.add_parser("ensure-ext", help="CREATE EXTENSION vector")
    sub.add_parser("init", help="Ensure embedding column/index for targets in elephant_gun.yaml")

    p_embed = sub.add_parser("embed", help="Backfill embeddings for a table or all targets")
    p_embed.add_argument("--table", help="target table name (default: all)")
    p_embed.add_argument("--batch", type=int, default=500)

    p_query = sub.add_parser("query", help="Hybrid query on a target table")
    p_query.add_argument("--table", required=True, help="target table name (must exist in config)")
    p_query.add_argument("--q", required=True, help="natural language text (semantic intent)")
    p_query.add_argument("--days", type=int, default=None)
    p_query.add_argument("--limit", type=int, default=20)
    p_query.add_argument("--dry-run", action="store_true")
    p_query.add_argument("--min-sim", type=float, default=None,
                         help="filter by minimum sim01 score (0..1); e.g., 0.70")

    # New: scan subcommand (heavy logic implemented in elephant_gun.eg_scan)
    p_scan = sub.add_parser("scan", help="Inspect DB schema and propose embedding text_template per table")
    p_scan.add_argument("--schema", default="public")
    p_scan.add_argument("--out", default="profiles/current/schema.yaml")
    p_scan.add_argument("--limit", type=int, default=5)

    args = ap.parse_args()

    # Commands that do not require config file
    if args.cmd == "ensure-ext":
        ensure_ext()
        return
    if args.cmd == "scan":
        # Handle scan BEFORE loading config (config may not exist yet)
        scan_schema(schema=args.schema, out_path=args.out, preview_limit=args.limit)
        return

    # Load config for commands that depend on it
    cfg = load_cfg()

    if args.cmd == "init":
        init_targets(cfg)
    elif args.cmd == "embed":
        backfill_embeddings(cfg, table=args.table, batch=args.batch)
    elif args.cmd == "query":
        hybrid_query(cfg, table=args.table, q=args.q, days=args.days,
                     limit=args.limit, dry_run=args.dry_run, min_sim=args.min_sim)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
