# Dataset wrappers module
import json
from typing import List, Dict, Any


def load_CUAD(path: str) -> List[Dict[str, Any]]:
    """
    Load CUAD evaluation dataset.
    
    Args:
        path: Path to cuad_eval.json file
        
    Returns:
        List of standardized examples with format:
        {
            "input": {"question": str, "context": str},
            "gold": str (gold answer span),
            "metadata": {"task": "CUAD", ...}
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append({
            "input": {
                "question": item.get("question", ""),
                "context": item.get("context", "")  # Optional context from paragraph
            },
            "gold": item.get("gold_answer", ""),
            "metadata": {
                "task": "CUAD",
                "has_context": bool(item.get("context"))
            }
        })
    
    return examples


def load_ECHR(path: str) -> List[Dict[str, Any]]:
    """
    Load ECHR evaluation dataset.
    
    Args:
        path: Path to echr_eval.json file
        
    Returns:
        List of standardized examples with format:
        {
            "input": {"case_text": str},
            "gold": List[str] (gold violated articles),
            "metadata": {"task": "ECHR", ...}
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append({
            "input": {
                "case_text": item.get("text", "")
            },
            "gold": item.get("gold_articles", []),  # List of article identifiers
            "metadata": {
                "task": "ECHR",
                "text_length": len(item.get("text", ""))
            }
        })
    
    return examples


def load_LEDGAR(path: str) -> List[Dict[str, Any]]:
    """
    Load LEDGAR evaluation dataset.
    
    Args:
        path: Path to ledgar_eval.json file
        
    Returns:
        List of standardized examples with format:
        {
            "input": {"clause_text": str},
            "gold": str (gold category label),
            "metadata": {"task": "LEDGAR", ...}
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append({
            "input": {
                "clause_text": item.get("text", "")
            },
            "gold": item.get("gold_label", ""),
            "metadata": {
                "task": "LEDGAR",
                "text_length": len(item.get("text", ""))
            }
        })
    
    return examples

