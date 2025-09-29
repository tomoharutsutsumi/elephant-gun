# Elephant Gun 🐘🔫
*A local-first CLI for hybrid SQL + semantic search on PostgreSQL.*

Elephant Gun lets you query your existing PostgreSQL tables with a mix of:
- **Structured filters (SQL)** — e.g., `created_at < 30 days`
- **Semantic search (pgvector + embeddings)** — e.g., “things that look like trouble”

No external APIs, no servers.  
Everything runs locally on your Postgres + Python.

## ✨ Features
- CLI commands for setup, embedding, and querying
- pgvector integration (cosine similarity search)
- Sentence-transformers embeddings
- Natural language + SQL filter fusion
- Local-first: your data never leaves your database

## 🚀 Quickstart

### 1. Clone & install
git clone https://github.com/<yourname>/elephant-gun.git
cd elephant-gun
python -m venv .venv && source .venv/bin/activate
pip install -e .

### 2. Start Postgres with pgvector
Using Docker:
docker run --name pg-vec -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -d pgvector/pgvector:pg14

export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/postgres"

### 3. Prepare database
# Enable extension
elephant-gun ensure-ext

# Initialize (embedding column + index)
elephant-gun init

# Create sample table
psql "$DATABASE_URL" <<'SQL'
CREATE TABLE IF NOT EXISTS tickets(
  id BIGSERIAL PRIMARY KEY,
  contract_id BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  title TEXT,
  body  TEXT
);
INSERT INTO tickets (contract_id, title, body) VALUES
(101, 'Refund dispute', 'Customer claims double charge and requests refund.'),
(102, 'Service outage report', 'Intermittent downtime observed by users on EU region.'),
(103, 'Feature request', 'User asks for export to CSV. No issue reported.'),
(104, 'Chargeback received', 'Bank notified a chargeback likely due to fraud suspicion.')
ON CONFLICT DO NOTHING;
SQL

### 4. Embed & query
elephant-gun embed --table tickets
elephant-gun query --table tickets --q "things that look like trouble" --days 30 --limit 10

## ⚙️ CLI Commands
elephant-gun ensure-ext        # Enable pgvector extension
elephant-gun init              # Add embedding column + index
elephant-gun embed --table T   # Embed rows into vectors
elephant-gun query --table T --q "text" [--days N --limit M --dry-run]

## 🛠 Requirements
- Python 3.9+
- PostgreSQL 14+ with pgvector
- Docker (optional, easiest way to start Postgres+pgvector)

## 📜 License
MIT — see [LICENSE](LICENSE)
