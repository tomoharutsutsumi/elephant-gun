#!/usr/bin/env python3
import os, sys, argparse, re
from typing import List, Optional
import yaml
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg

DB_URL = os.getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/postgres")

# ---------- Config models ----------
class Target(BaseModel):
    table: str
    key: str
    time_column: Optional[str] = None
    text_template: str
    filters: List[str] = []

class Config(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    targets: List[Target]

# ---------- Config I/O ----------
def load_cfg(path="elephant_gun.yaml") -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

# ---------- DB helpers ----------
def conn():
    return psycopg.connect(DB_URL, autocommit=True)

def ensure_ext():
    with conn() as con:
        con.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("vector extension ensured.")

def ensure_target_schema(t: Target, embed_dim: int):
    with conn() as con:
        con.execute(f"ALTER TABLE {t.table} ADD COLUMN IF NOT EXISTS embedding VECTOR({embed_dim});")
        if t.time_column:
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{t.table}_{t.time_column} ON {t.table} ({t.time_column} DESC);")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{t.table}_{t.key} ON {t.table} ({t.key});")

# ---------- Embedding ----------
_model = None
def get_model(cfg: Config):
    global _model
    if _model is None:
        _model = SentenceTransformer(cfg.model)
    return _model

def render_text(row: dict, template: str) -> str:
    out = template
    for k, v in row.items():
        out = out.replace("{{"+k+"}}", "" if v is None else str(v))
    return re.sub(r"\s+", " ", out).strip()

def iter_batch_rows_to_embed(t: Target, batch: int = 500):
    cols_needed = set([t.key]) | set(re.findall(r"{{\s*([a-zA-Z0-9_]+)\s*}}", t.text_template))
    select_cols = ", ".join(sorted(cols_needed))
    with conn() as con:
        while True:
            rows = con.execute(
                f"SELECT {select_cols} FROM {t.table} WHERE embedding IS NULL LIMIT %s", (batch,)
            ).fetchall()
            if not rows:
                break
            names = [d.name for d in con.execute(f"SELECT {select_cols} FROM {t.table} LIMIT 0").description]
            dicts = [dict(zip(names, r)) for r in rows]
            yield dicts

def backfill_embeddings(cfg: Config, table: Optional[str] = None, batch: int = 500):
    m = get_model(cfg)
    targets = [t for t in cfg.targets if (table is None or t.table == table)]
    for t in targets:
        ensure_target_schema(t, cfg.embed_dim)
        total = 0
        while True:
            chunk = next(iter_batch_rows_to_embed(t, batch), [])
            if not chunk: break
            texts = [render_text(r, t.text_template) for r in chunk]
            vecs = m.encode(texts, normalize_embeddings=True)
            with conn() as con:
                with con.cursor() as cur:
                    for r, v in zip(chunk, vecs):
                        vec_list = ",".join(f"{x:.7f}" for x in v.tolist())
                        cur.execute(
                            f"UPDATE {t.table} SET embedding='[{vec_list}]'::vector WHERE {t.key}=%s",
                            (r[t.key],)
                        )
            total += len(chunk)
            print(f"[{t.table}] embedded +{len(chunk)} (total {total})")
        print(f"[{t.table}] embedding done: {total}")

def embed_query(cfg: Config, text: str):
    m = get_model(cfg)
    return m.encode(text, normalize_embeddings=True)

# ---------- Query ----------
def sql_filters(t: Target, days: Optional[int]) -> str:
    parts = []
    for f in t.filters:
        parts.append(f"({f})")
    if days and t.time_column:
        parts.append(f"{t.time_column} >= now() - interval '{int(days)} days'")
    return " AND ".join(parts) if parts else "TRUE"

def hybrid_query(cfg: Config, table: str, q: str, days: Optional[int], limit: int, dry_run: bool):
    qvec = embed_query(cfg, q)
    vec_list = ",".join(f"{x:.7f}" for x in qvec.tolist())

    matches = [x for x in cfg.targets if x.table == table]
    if not matches:
        print(f"target table not found in elephant_gun.yaml: {table}", file=sys.stderr)
        sys.exit(1)
    t = matches[0]

    where = sql_filters(t, days)
    preview = "left(coalesce(title,'')||' '||coalesce(body,''), 120) as preview" if t.table=="tickets" else "NULL as preview"

    sql = f"""
    SELECT {t.key} AS id, {preview},
        1 - (embedding <=> '[{vec_list}]'::vector) AS cosine_sim,       -- [-1, 1]
        1 - 0.5 * (embedding <=> '[{vec_list}]'::vector) AS sim01       -- [0, 1] 表示用
    FROM {t.table}
    WHERE {where} AND embedding IS NOT NULL
    ORDER BY embedding <=> '[{vec_list}]'::vector
    LIMIT {int(limit)};
    """.strip()

    if dry_run:
        print(sql)
        return

    with conn() as con:
        rows = con.execute(sql).fetchall()
    for r in rows:
        rid, prev, sim, sim01 = r
        print(f"[{rid}] sim01={sim01:.3f}  {prev}")

def init_targets(cfg: Config):
    for t in cfg.targets:
        ensure_target_schema(t, cfg.embed_dim)
    print("init done. (embedding column/index ensured)")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(prog="elephant-gun", description="Hybrid SQL + semantic search CLI for existing Postgres.")
    sub = ap.add_subparsers(dest="cmd", required=True)

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

    args = ap.parse_args()

    if args.cmd == "ensure-ext":
        ensure_ext()
        return

    cfg = load_cfg()

    if args.cmd == "init":
        init_targets(cfg)
    elif args.cmd == "embed":
        backfill_embeddings(cfg, table=args.table, batch=args.batch)
    elif args.cmd == "query":
        hybrid_query(cfg, table=args.table, q=args.q, days=args.days, limit=args.limit, dry_run=args.dry_run)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
