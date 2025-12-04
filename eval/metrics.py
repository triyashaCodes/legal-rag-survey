# Metrics module for legal RAG evaluation

from typing import List, Dict, Set, Tuple, Any
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import re


# ============================================================================
# CUAD Metrics: Span-level F1
# ============================================================================

def normalize_span(text: str) -> str:
    """Normalize text span for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def compute_span_f1(predicted_spans: List[str], gold_spans: List[str]) -> Dict[str, float]:
    """
    Compute span-level F1 metrics for CUAD task.
    
    Args:
        predicted_spans: List of predicted answer spans
        gold_spans: List of gold answer spans
        
    Returns:
        Dictionary with exact_match_f1, partial_f1, precision, recall
    """
    if not gold_spans and not predicted_spans:
        return {"exact_match_f1": 1.0, "partial_f1": 1.0, "precision": 1.0, "recall": 1.0}
    
    if not gold_spans:
        return {"exact_match_f1": 0.0, "partial_f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    if not predicted_spans:
        return {"exact_match_f1": 0.0, "partial_f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Normalize spans
    pred_normalized = [normalize_span(s) for s in predicted_spans]
    gold_normalized = [normalize_span(s) for s in gold_spans]
    
    # Exact match: spans must match exactly
    pred_set = set(pred_normalized)
    gold_set = set(gold_normalized)
    
    exact_tp = len(pred_set & gold_set)
    exact_precision = exact_tp / len(pred_set) if pred_set else 0.0
    exact_recall = exact_tp / len(gold_set) if gold_set else 0.0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if (exact_precision + exact_recall) > 0 else 0.0
    
    # Partial match: check if predicted span contains or is contained in gold span
    partial_tp = 0
    for pred in pred_normalized:
        for gold in gold_normalized:
            if pred in gold or gold in pred:
                partial_tp += 1
                break
    
    partial_precision = partial_tp / len(pred_normalized) if pred_normalized else 0.0
    partial_recall = partial_tp / len(gold_normalized) if gold_normalized else 0.0
    partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall) if (partial_precision + partial_recall) > 0 else 0.0
    
    return {
        "exact_match_f1": exact_f1,
        "partial_f1": partial_f1,
        "precision": exact_precision,
        "recall": exact_recall
    }


def parse_spans_from_text(text: str) -> List[str]:
    """
    Parse span predictions from model output text.
    
    Handles various formats:
    - "span1, span2, span3"
    - "span1\nspan2\nspan3"
    - JSON-like lists
    - Single span
    
    Args:
        text: Model output text
        
    Returns:
        List of extracted spans
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Try JSON list format
    try:
        import json
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed if s]
    except:
        pass
    
    # Try comma-separated
    if "," in text:
        spans = [s.strip() for s in text.split(",")]
        return [s for s in spans if s]
    
    # Try newline-separated
    if "\n" in text:
        spans = [s.strip() for s in text.split("\n")]
        return [s for s in spans if s]
    
    # Single span
    return [text] if text else []


# ============================================================================
# ECHR Metrics: Violation F1 + Evidence Precision/Recall
# ============================================================================

def parse_violations_from_text(text: str) -> List[str]:
    """
    Parse ECHR article violations from model output.
    
    Handles formats like:
    - "Article 6, Article 8"
    - "Article 6\nArticle 8"
    - "None"
    - JSON lists
    
    Args:
        text: Model output text
        
    Returns:
        List of article identifiers (e.g., ["Article 6", "Article 8"])
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Check for "None" or "no violations"
    if text.lower() in ["none", "no violations", "no violation", "n/a", "na"]:
        return []
    
    # Try JSON list format
    try:
        import json
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed if a]
    except:
        pass
    
    # Extract article patterns (e.g., "Article 6", "Art. 8", "Article 6(1)")
    article_pattern = r'Article\s+\d+[^\s,]*|Art\.\s*\d+[^\s,]*'
    articles = re.findall(article_pattern, text, re.IGNORECASE)
    
    if articles:
        # Normalize format
        normalized = []
        for art in articles:
            # Normalize to "Article X" format
            art = re.sub(r'Art\.\s*', 'Article ', art, flags=re.IGNORECASE)
            normalized.append(art.strip())
        return list(set(normalized))  # Remove duplicates
    
    # Fallback: split by comma or newline
    if "," in text:
        items = [s.strip() for s in text.split(",")]
        return [s for s in items if s and s.lower() != "none"]
    
    if "\n" in text:
        items = [s.strip() for s in text.split("\n")]
        return [s for s in items if s and s.lower() != "none"]
    
    return [text] if text else []


def compute_violation_f1(predicted_violations: List[str], gold_violations: List[str]) -> Dict[str, float]:
    """
    Compute violation F1 metrics for ECHR task.
    
    Args:
        predicted_violations: List of predicted violated articles
        gold_violations: List of gold violated articles
        
    Returns:
        Dictionary with f1, precision, recall, and per-article metrics
    """
    # Normalize article identifiers
    pred_set = set(normalize_span(v) for v in predicted_violations)
    gold_set = set(normalize_span(v) for v in gold_violations)
    
    # Compute metrics
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def compute_evidence_metrics(
    predicted_evidence: List[str],
    gold_evidence: List[str],
    case_text: str
) -> Dict[str, float]:
    """
    Compute evidence precision/recall for ECHR task.
    
    This measures how well the model grounds its predictions in the case text.
    
    Args:
        predicted_evidence: List of predicted evidence spans
        gold_evidence: List of gold evidence spans
        case_text: Full case text for context
        
    Returns:
        Dictionary with evidence precision, recall, f1
    """
    if not gold_evidence and not predicted_evidence:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if not gold_evidence:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not predicted_evidence:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Normalize evidence spans
    pred_normalized = [normalize_span(e) for e in predicted_evidence]
    gold_normalized = [normalize_span(e) for e in gold_evidence]
    
    # Check overlap (partial match)
    tp = 0
    for pred in pred_normalized:
        for gold in gold_normalized:
            # Check if spans overlap (one contains the other or they share significant text)
            if pred in gold or gold in pred or len(set(pred.split()) & set(gold.split())) > 3:
                tp += 1
                break
    
    precision = tp / len(pred_normalized) if pred_normalized else 0.0
    recall = tp / len(gold_normalized) if gold_normalized else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================================
# LEDGAR Metrics: Macro F1 + Calibration
# ============================================================================

def compute_macro_f1(predicted_labels: List[str], gold_labels: List[str]) -> Dict[str, Any]:
    """
    Compute macro F1 and per-class metrics for LEDGAR task.
    
    Args:
        predicted_labels: List of predicted category labels
        gold_labels: List of gold category labels
        
    Returns:
        Dictionary with macro_f1, weighted_f1, per_class_metrics, confusion_matrix
    """
    # Get all unique labels
    all_labels = sorted(set(predicted_labels + gold_labels))
    
    if not all_labels:
        return {
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_f1": {},
            "confusion_matrix": []
        }
    
    # Compute macro F1
    macro_f1 = f1_score(gold_labels, predicted_labels, average="macro", zero_division=0.0)
    weighted_f1 = f1_score(gold_labels, predicted_labels, average="weighted", zero_division=0.0)
    micro_f1 = f1_score(gold_labels, predicted_labels, average="micro", zero_division=0.0)
    
    # Per-class metrics
    per_class_f1 = {}
    precision_per_class = precision_score(
        gold_labels, predicted_labels, average=None, labels=all_labels, zero_division=0.0
    )
    recall_per_class = recall_score(
        gold_labels, predicted_labels, average=None, labels=all_labels, zero_division=0.0
    )
    f1_per_class = f1_score(
        gold_labels, predicted_labels, average=None, labels=all_labels, zero_division=0.0
    )
    
    for i, label in enumerate(all_labels):
        per_class_f1[label] = {
            "f1": float(f1_per_class[i]),
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i])
        }
    
    # Confusion matrix
    cm = confusion_matrix(gold_labels, predicted_labels, labels=all_labels)
    
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(micro_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "all_labels": all_labels
    }


def compute_calibration_metrics(
    predicted_labels: List[str],
    gold_labels: List[str],
    predicted_probs: List[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute calibration metrics for LEDGAR classification.
    
    Calibration measures how well predicted probabilities match actual frequencies.
    
    Args:
        predicted_labels: List of predicted labels
        gold_labels: List of gold labels
        predicted_probs: Optional list of probability distributions (one per prediction)
        
    Returns:
        Dictionary with calibration error, Brier score, etc.
    """
    if predicted_probs is None:
        # If no probabilities provided, return basic metrics
        return {
            "calibration_error": None,
            "brier_score": None,
            "note": "No probability predictions provided"
        }
    
    # Compute calibration error (ECE - Expected Calibration Error)
    # Group predictions into bins by confidence
    n_bins = 10
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        bin_indices = []
        for j, probs in enumerate(predicted_probs):
            pred_label = predicted_labels[j]
            confidence = probs.get(pred_label, 0.0)
            if bin_lower <= confidence < bin_upper:
                bin_indices.append(j)
        
        if not bin_indices:
            continue
        
        # Compute accuracy and average confidence for this bin
        bin_correct = sum(1 for j in bin_indices if predicted_labels[j] == gold_labels[j])
        bin_accuracy = bin_correct / len(bin_indices)
        bin_confidence = sum(predicted_probs[j].get(predicted_labels[j], 0.0) for j in bin_indices) / len(bin_indices)
        
        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_confidence)
        bin_counts.append(len(bin_indices))
    
    # Expected Calibration Error
    total_samples = sum(bin_counts)
    if total_samples > 0:
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        )
    else:
        ece = 0.0
    
    # Brier score (mean squared error of probabilities)
    brier_scores = []
    for j, probs in enumerate(predicted_probs):
        pred_label = predicted_labels[j]
        gold_label = gold_labels[j]
        
        # One-hot encoding for gold label
        for label in probs.keys():
            prob = probs[label]
            actual = 1.0 if label == gold_label else 0.0
            brier_scores.append((prob - actual) ** 2)
    
    brier_score = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0
    
    return {
        "calibration_error": float(ece),
        "brier_score": float(brier_score),
        "n_bins": n_bins,
        "bin_accuracies": [float(a) for a in bin_accuracies],
        "bin_confidences": [float(c) for c in bin_confidences]
    }


# ============================================================================
# Utility Functions
# ============================================================================

def compute_bal_acc(true_labels, predicted_labels):
    """Compute balanced accuracy (backward compatibility)."""
    return balanced_accuracy_score(true_labels, predicted_labels)


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple examples.
    
    Args:
        metrics_list: List of metric dictionaries (one per example)
        
    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}
    
    # Average all numeric values
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m.get(key) for m in metrics_list if m.get(key) is not None]
        if values:
            aggregated[key] = sum(values) / len(values)
    
    return aggregated
