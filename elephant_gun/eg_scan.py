# elephant_gun/eg_scan.py
from __future__ import annotations
import os
import re
import yaml
from typing import Optional
from .db import conn  # <-- if you already have a helper; else import from your main

def scan_schema(schema: str = "public",
                out_path: str = "profiles/current/schema.yaml",
                preview_limit: int = 5) -> None:
    """
    Inspect the DB schema and propose embedding targets per table:
      - key column (primary key or 'id')
      - time column (created_at/updated_at/etc.)
      - text columns (TEXT/VARCHAR)
      - fallback pseudo text_template when no text columns exist
    Writes a YAML profile to `out_path` and prints a small preview.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _is_text(dtype: str) -> bool:
        dtype = (dtype or "").lower()
        return ("text" in dtype) or ("character varying" in dtype) or ("varchar" in dtype) or ("citext" in dtype)

    def _is_time(dtype: str) -> bool:
        dtype = (dtype or "").lower()
        return ("timestamp" in dtype) or ("date" in dtype) or ("timestamptz" in dtype)

    TIME_NAME_RANK = [
        "created_at", "inserted_at", "createdon", "created_time", "event_time",
        "updated_at", "timestamp", "ts", "date", "time"
    ]

    PSEUDO_CAND_NAME_PAT = re.compile(
        r"(name|title|status|type|plan|region|country|city|category|email|phone|state|stage|priority|severity|reason|source|channel|method|brand|model)$",
        re.I
    )

    with conn() as con:
        tables = con.execute("""
            SELECT t.table_schema, t.table_name
            FROM information_schema.tables t
            WHERE t.table_type='BASE TABLE' AND t.table_schema=%s
            ORDER BY t.table_name
        """, (schema,)).fetchall()

        cols = con.execute("""
            SELECT c.table_name, c.column_name, c.data_type, c.is_nullable
            FROM information_schema.columns c
            WHERE c.table_schema=%s
            ORDER BY c.table_name, c.ordinal_position
        """, (schema,)).fetchall()

        pks = con.execute("""
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema=%s AND tc.constraint_type='PRIMARY KEY'
        """, (schema,)).fetchall()

    colmap = {}
    for tname, cname, dtype, _null in cols:
        colmap.setdefault(tname, []).append({"name": cname, "dtype": dtype})

    pkmap = {}
    for tname, cname in pks:
        pkmap.setdefault(tname, set()).add(cname)

    targets = []
    for _schema, tname in tables:
        tcols = colmap.get(tname, [])
        names = [c["name"] for c in tcols]

        # key
        key = None
        if tname in pkmap and pkmap[tname]:
            key = list(pkmap[tname])[0]
        elif "id" in names:
            key = "id"
        elif names:
            key = names[0]

        # time
        time_candidates = [c for c in tcols if _is_time(c["dtype"])]
        time = None
        if time_candidates:
            ranked = sorted(
                time_candidates,
                key=lambda c: (TIME_NAME_RANK.index(c["name"]) if c["name"] in TIME_NAME_RANK else 999)
            )
            time = ranked[0]["name"]

        # text columns
        text_cols = [c["name"] for c in tcols if _is_text(c["dtype"])]

        # template
        if text_cols:
            preferred_order = ["title", "name", "subject", "summary", "description", "body", "message", "notes"]
            ordered = sorted(text_cols, key=lambda n: (preferred_order.index(n) if n in preferred_order else 999, n))
            top = ordered[:4]
            tmpl = " || ' ' || ".join([f"coalesce({n}::text,'')" for n in top])
        else:
            cat_like = [c["name"] for c in tcols if PSEUDO_CAND_NAME_PAT.search(c["name"])]
            num_like = [n for n in names if re.search(r"(amount|price|total|count|score|rating|quantity|qty|revenue)$", n, re.I)]
            pick = (cat_like[:6] + num_like[:2])[:8] or names[:3]
            tmpl = " || ' ' || ".join([f"coalesce({n}::text,'')" for n in pick])

        targets.append({
            "table": tname,
            "key": key,
            "time_column": time,
            "text_template": tmpl,
        })

    profile = {"version": 1, "schema": schema, "targets": targets}
    with open(out_path, "w") as f:
        yaml.safe_dump(profile, f, sort_keys=False)

    print(f"wrote: {out_path}")
    for t in targets:
        print(f"- {t['table']}: key={t['key']} time={t['time_column']} text_template= {t['text_template']}")

    # Optional: preview first 1–2 tables
    try:
        with conn() as con:
            for t in targets[:2]:
                q = f"SELECT {t['key']} AS id, left({t['text_template']}, 120) AS preview FROM {t['table']} LIMIT {preview_limit};"
                rows = con.execute(q).fetchall()
                print(f"\nPreview {t['table']} (first {len(rows)} rows):")
                for r in rows:
                    print(f"  [{r[0]}] {r[1]}")
    except Exception as e:
        print(f"(preview failed: {e})")
