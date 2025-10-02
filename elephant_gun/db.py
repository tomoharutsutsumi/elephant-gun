# elephant_gun/db.py
import os
import psycopg

DB_URL = os.getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/postgres")

def conn():
    # Autocommit for utility-style operations
    return psycopg.connect(DB_URL, autocommit=True)
