import os, csv
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import psycopg

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_CSV   = os.path.join("data", "pairs.csv")
OUT_DIR    = os.path.join("models", "eg-miniLM-finetuned")
BATCH_SIZE = 32
EPOCHS     = 5            
WARMUP_STEPS = 100
DB_URL = os.environ.get("DATABASE_URL")

def load_pairs(path):
    examples = []
    with open(path) as f:
        reader = csv.DictReader(f)  # DictReader skips the header
        for row in reader:
            q, t = row["query"], row["text"]
            y = float(row["label"])
            examples.append(InputExample(texts=[q, t], label=y))
    return examples

def clear_embeddings():
    """Reset embeddings column to NULL so we can re-embed with the new fine-tuned model."""
    if not DB_URL:
        print("⚠️ DATABASE_URL is not set, skipping embedding reset.")
        return
    with psycopg.connect(DB_URL, autocommit=True) as con:
        con.execute("UPDATE tickets SET embedding = NULL;")
        print("✅ Reset tickets.embedding to NULL")

def main():
    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    train_examples = load_pairs(DATA_CSV)
    print(f"loaded {len(train_examples)} training pairs from {DATA_CSV}")

    model = SentenceTransformer(BASE_MODEL)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True,
        output_path=OUT_DIR
    )
    print(f"saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
    clear_embeddings()