import os
import json
from datasets import load_dataset
from tqdm import tqdm
from random import seed, shuffle

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Evaluation split fraction
EVAL_FRAC = 0.2
RANDOM_SEED = 42

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def chunk_text(text, max_words=200):
    """Chunk text into segments of max_words each"""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# -----------------------------
# CUAD
# -----------------------------
def prepare_CUAD():
    print("\n=== Preparing CUAD ===")
    cuad_dir = os.path.join(DATA_DIR, "CUAD")
    ensure_dir(cuad_dir)

    contracts_dir = os.path.join(cuad_dir, "full_contract_txt")
    qas_path = os.path.join(cuad_dir, "CUAD_v1.json")

    train_chunks_path = os.path.join(cuad_dir, "cuad_chunks_train.jsonl")
    eval_path = os.path.join(cuad_dir, "cuad_eval.json")

    # -----------------------------
    # Process Q&A data for evaluation
    # -----------------------------
    eval_contracts = set()
    if os.path.exists(qas_path):
        with open(qas_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        # Random 20% of contracts for evaluation
        from random import seed, sample
        seed(42)
        all_contracts = [example.get("title") + ".txt" for example in qa_data["data"]]
        eval_contracts = set(sample(all_contracts, int(len(all_contracts) * 0.2)))

        eval_examples = []
        for example in qa_data["data"]:
            contract_file = example.get("title") + ".txt"
            if contract_file in eval_contracts:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"]
                    for q in paragraph["qas"]:
                        gold_answer = q["answers"][0]["text"] if q.get("answers") else ""
                        eval_examples.append({
                            "question": q["question"],
                            "gold_answer": gold_answer,
                            "context": context
                        })

        # Save CUAD eval examples
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_examples, f, indent=2)
        print(f"[CUAD] Eval Q&A saved: {len(eval_examples)} → {eval_path}")
    else:
        print(f"[CUAD] Q&A file not found: {qas_path}")

    # -----------------------------
    # Chunk training contracts (not in eval)
    # -----------------------------
    if os.path.exists(contracts_dir):
        train_chunks = []
        txt_files = [f for f in os.listdir(contracts_dir) if f.endswith(".txt")]
        for txt_file in tqdm(txt_files, desc="Chunking CUAD train contracts"):
            if txt_file in eval_contracts:
                continue
            txt_path = os.path.join(contracts_dir, txt_file)
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if text.strip():
                        for chunk in chunk_text(text):
                            train_chunks.append({"text": chunk})
            except Exception as e:
                print(f"Warning: Could not process {txt_file}: {e}")
                continue

        # Save train chunks
        with open(train_chunks_path, "w", encoding="utf-8") as f:
            for c in train_chunks:
                f.write(json.dumps(c) + "\n")
        print(f"[CUAD] Train chunks saved: {len(train_chunks)} → {train_chunks_path}")
    else:
        print(f"[CUAD] Contracts directory not found: {contracts_dir}")

# -----------------------------
# LEDGAR
# -----------------------------
def prepare_LEDGAR():
    print("\n=== Preparing LEDGAR ===")
    ledgar_dir = os.path.join(DATA_DIR, "LEDGAR")
    ensure_dir(ledgar_dir)

    try:
        ds = load_dataset("lex_glue", "ledgar")
        train_data = ds["train"]
        total = len(train_data)
        split_size = int(total * EVAL_FRAC)

        # Shuffle and split
        seed(RANDOM_SEED)
        indices = list(range(total))
        shuffle(indices)
        eval_indices = set(indices[:split_size])
        train_indices = indices[split_size:]

        train_chunks = []
        eval_examples = []

        for i, item in enumerate(tqdm(train_data, desc="Processing LEDGAR")):
            text = item.get("text", "")
            label = item.get("label", "")
            if not text:
                continue

            example = {"text": text, "gold_label": label}
            if i in eval_indices:
                eval_examples.append(example)
            else:
                for chunk in chunk_text(text):
                    train_chunks.append({"text": chunk})

        # Save
        train_chunks_path = os.path.join(ledgar_dir, "ledgar_chunks_train.jsonl")
        eval_path = os.path.join(ledgar_dir, "ledgar_eval.json")

        with open(train_chunks_path, "w", encoding="utf-8") as f:
            for c in train_chunks:
                f.write(json.dumps(c) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_examples, f, indent=2)

        print(f"[LEDGAR] Train chunks: {len(train_chunks)} → {train_chunks_path}")
        print(f"[LEDGAR] Eval examples: {len(eval_examples)} → {eval_path}")

    except Exception as e:
        print(f"[LEDGAR] Could not load dataset: {e}")

# -----------------------------
# ECHR
# -----------------------------
def prepare_ECHR():
    print("\n=== Preparing ECHR ===")
    echr_dir = os.path.join(DATA_DIR, "ECHR")
    ensure_dir(echr_dir)

    try:
        ds = load_dataset("lex_glue", "ecthr_a")
        total = len(ds["train"])
        split_size = int(total * EVAL_FRAC)

        seed(RANDOM_SEED)
        indices = list(range(total))
        shuffle(indices)
        eval_indices = set(indices[:split_size])
        train_indices = indices[split_size:]

        train_chunks = []
        eval_examples = []

        for i, item in enumerate(tqdm(ds["train"], desc="Processing ECHR")):
            text_list = item.get("text", [])
            full_text = " ".join(text_list) if isinstance(text_list, list) else str(text_list)
            if not full_text.strip():
                continue

            example = {
                "text": full_text,
                "gold_articles": item.get("labels", [])
            }

            if i in eval_indices:
                eval_examples.append(example)
            else:
                for chunk in chunk_text(full_text):
                    train_chunks.append({"text": chunk})

        # Save
        train_chunks_path = os.path.join(echr_dir, "echr_chunks_train.jsonl")
        eval_path = os.path.join(echr_dir, "echr_eval.json")

        with open(train_chunks_path, "w", encoding="utf-8") as f:
            for c in train_chunks:
                f.write(json.dumps(c) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_examples, f, indent=2)

        print(f"[ECHR] Train chunks: {len(train_chunks)} → {train_chunks_path}")
        print(f"[ECHR] Eval examples: {len(eval_examples)} → {eval_path}")

    except Exception as e:
        print(f"[ECHR] Could not load dataset: {e}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    prepare_CUAD()
    prepare_LEDGAR()
    prepare_ECHR()
    print("\nAll datasets prepared with train/eval split!")
