# Data Directory

This directory contains the legal datasets used in the RAG survey.

## Datasets

- **CUAD/**: Contract Understanding Atticus Dataset
- **ECHR/**: European Court of Human Rights case law
- **LEDGAR/**: Legal Dataset for Agreement Recognition

## Note

Large data files (JSONL, JSON, PDF, TXT, XLSX, CSV) are excluded from git via `.gitignore`. 
To use the datasets, download and prepare them using:

```bash
python utils/download_datasets.py
```

This will generate the necessary chunk files for indexing.

