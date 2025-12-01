import os
import json
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

# Get project root directory (parent of utils/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ---------------------------------------------------------------------
# Utility: ensure folder exists
# ---------------------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ---------------------------------------------------------------------
# Text chunker for long legal paragraphs
# ---------------------------------------------------------------------
def chunk_text(text, max_words=200):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])


# ---------------------------------------------------------------------
# CUAD
# ---------------------------------------------------------------------
def download_and_prepare_CUAD():
    print("\n=== Preparing CUAD from local data ===")

    cuad_dir = os.path.join(DATA_DIR, "CUAD")
    ensure_dir(cuad_dir)
    contracts_dir = os.path.join(cuad_dir, "full_contract_txt")
    qas_path = os.path.join(cuad_dir, "CUAD_v1.json")
    sample_output_path = os.path.join(cuad_dir, "sample.json")
    chunks_output_path = os.path.join(cuad_dir, "cuad_chunks.jsonl")

    # Process full contract text files to create chunks for RAG
    if os.path.exists(contracts_dir):
        chunks = []
        txt_files = [f for f in os.listdir(contracts_dir) if f.endswith('.txt')]
        
        for txt_file in tqdm(txt_files, desc="Chunking CUAD contracts"):
            txt_path = os.path.join(contracts_dir, txt_file)
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if text.strip():
                        for chunk in chunk_text(text):
                            chunks.append({"text": chunk})
            except Exception as e:
                print(f"Warning: Could not process {txt_file}: {e}")
                continue
        
        # Write chunks to JSONL file
        with open(chunks_output_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        
        print(f"CUAD chunks prepared → {chunks_output_path} ({len(chunks)} chunks)")
    else:
        print(f"Warning: Contracts directory not found: {contracts_dir}")

    # Process QA data (CUAD gold questions/answers) for evaluation
    if os.path.exists(qas_path):
        try:
            with open(qas_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            examples = []

            for example in qa_data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"]
                    for q in paragraph["qas"]:
                        question = q["question"]
                        # There may be multiple answers, take first
                        if q.get("answers"):
                            gold_answer = q["answers"][0]["text"]
                        else:
                            gold_answer = ""
                        examples.append({
                            "question": question,
                            "gold_answer": gold_answer,
                            "context": context
                        })

            with open(sample_output_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2)

            print(f"CUAD Q&A processed → {sample_output_path} ({len(examples)} examples)")
        except Exception as e:
            print(f"Warning: Could not process Q&A data: {e}")
    else:
        print(f"Warning: Q&A file not found: {qas_path}")

# ---------------------------------------------------------------------
# LEDGAR
# ---------------------------------------------------------------------
def download_and_prepare_LEDGAR():
    print("\n=== Downloading LEDGAR ===")
    try:
        ds = load_dataset("lex_glue", "ledgar")
    except Exception as e:
        print(f"LEDGAR dataset could not be loaded: {e}. Skipping.")
        return

    ledgar_dir = os.path.join(DATA_DIR, "LEDGAR")
    ensure_dir(ledgar_dir)

    chunks = []
    for item in tqdm(ds["train"], desc="Chunking LEDGAR"):
        text = item.get("text", "")
        if text:
            for chunk in chunk_text(text):
                chunks.append({"text": chunk})

    output_path = os.path.join(ledgar_dir, "ledgar_chunks.jsonl")
    with open(output_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"LEDGAR prepared → {output_path} ({len(chunks)} chunks)")


# ---------------------------------------------------------------------
# ECHR (Case Law)
# ---------------------------------------------------------------------
def download_and_prepare_ECHR():
    print("\n=== Downloading ECHR ===")
    try:
        ds = load_dataset("lex_glue", "ecthr_a")
    except Exception as e:
        print(f"ECHR dataset could not be loaded: {e}. Skipping.")
        return

    echr_dir = os.path.join(DATA_DIR, "ECHR")
    ensure_dir(echr_dir)

    chunks = []
    for item in tqdm(ds["train"], desc="Chunking ECHR"):
        # ECHR has "text" field which is a list of strings
        text_list = item.get("text", [])
        if text_list:
            # Join all text segments into one document
            full_text = " ".join(text_list) if isinstance(text_list, list) else str(text_list)
            if full_text:
                for chunk in chunk_text(full_text):
                    chunks.append({"text": chunk})

    output_path = os.path.join(echr_dir, "echr_chunks.jsonl")
    with open(output_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"ECHR prepared → {output_path} ({len(chunks)} chunks)")

# ---------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------
if __name__ == "__main__":
    success_count = 0
    total_count = 3
    
    try:
        download_and_prepare_CUAD()
        success_count += 1
    except Exception as e:
        print(f"Error processing CUAD: {e}")
    
    try:
        download_and_prepare_LEDGAR()
        success_count += 1
    except Exception as e:
        print(f"Error processing LEDGAR: {e}")
    
    try:
        download_and_prepare_ECHR()
        success_count += 1
    except Exception as e:
        print(f"Error processing ECHR: {e}")

    print(f"\nProcessed {success_count}/{total_count} datasets successfully!")
    if success_count > 0:
        print("You are ready to build FAISS indexes and run orchestration benchmarks.")
    else:
        print("No datasets were successfully processed. Please check dataset availability and network connection.")
