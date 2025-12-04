# Task-specific evaluators for CUAD, ECHR, and LEDGAR

from typing import List, Dict, Any
from tqdm import tqdm
from orchestrators.base_agent import BaseLegalAgent
from eval.metrics import (
    compute_span_f1,
    parse_spans_from_text,
    compute_violation_f1,
    parse_violations_from_text,
    compute_evidence_metrics,
    compute_macro_f1,
    compute_calibration_metrics
)


class CUADEvaluator:
    """Evaluator for CUAD span extraction task."""
    
    def __init__(self, agent: BaseLegalAgent):
        """
        Initialize CUAD evaluator.
        
        Args:
            agent: Agent implementing extract_spans() method
        """
        self.agent = agent
    
    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate agent on CUAD dataset.
        
        Args:
            dataset: List of examples with format:
                {
                    "input": {"question": str, "context": Optional[str]},
                    "gold": str (gold answer span),
                    "metadata": {...}
                }
        
        Returns:
            Dictionary with metrics and predictions
        """
        predictions = []
        gold_spans = []
        all_metrics = []
        
        for example in tqdm(dataset, desc="Evaluating CUAD"):
            input_data = example["input"]
            question = input_data["question"]
            context = input_data.get("context")
            gold = example["gold"]
            
            # Get prediction
            pred_text = self.agent.extract_spans(question, context)
            pred_spans = parse_spans_from_text(pred_text)
            
            # Gold span as list (single span)
            gold_span_list = [gold] if gold else []
            
            # Compute metrics
            metrics = compute_span_f1(pred_spans, gold_span_list)
            all_metrics.append(metrics)
            
            predictions.append({
                "question": question,
                "predicted": pred_text,
                "predicted_spans": pred_spans,
                "gold": gold
            })
            gold_spans.append(gold_span_list)
        
        # Aggregate metrics
        avg_metrics = {
            "exact_match_f1": sum(m["exact_match_f1"] for m in all_metrics) / len(all_metrics),
            "partial_f1": sum(m["partial_f1"] for m in all_metrics) / len(all_metrics),
            "precision": sum(m["precision"] for m in all_metrics) / len(all_metrics),
            "recall": sum(m["recall"] for m in all_metrics) / len(all_metrics)
        }
        
        return {
            "metrics": avg_metrics,
            "per_example_metrics": all_metrics,
            "predictions": predictions,
            "n_examples": len(dataset)
        }


class ECHREvaluator:
    """Evaluator for ECHR violation prediction task."""
    
    def __init__(self, agent: BaseLegalAgent):
        """
        Initialize ECHR evaluator.
        
        Args:
            agent: Agent implementing predict_violations() method
        """
        self.agent = agent
    
    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate agent on ECHR dataset.
        
        Args:
            dataset: List of examples with format:
                {
                    "input": {"case_text": str},
                    "gold": List[str] (gold violated articles),
                    "metadata": {...}
                }
        
        Returns:
            Dictionary with metrics and predictions
        """
        predictions = []
        all_metrics = []
        
        for example in tqdm(dataset, desc="Evaluating ECHR"):
            input_data = example["input"]
            case_text = input_data["case_text"]
            gold_violations = example["gold"]
            
            # Get prediction
            pred_violations = self.agent.predict_violations(case_text)
            
            # Compute violation F1
            metrics = compute_violation_f1(pred_violations, gold_violations)
            all_metrics.append(metrics)
            
            predictions.append({
                "case_text": case_text[:200],  # Store first 200 chars for reference
                "predicted_violations": pred_violations,
                "gold_violations": gold_violations
            })
        
        # Aggregate metrics
        avg_metrics = {
            "f1": sum(m["f1"] for m in all_metrics) / len(all_metrics),
            "precision": sum(m["precision"] for m in all_metrics) / len(all_metrics),
            "recall": sum(m["recall"] for m in all_metrics) / len(all_metrics),
            "avg_tp": sum(m["tp"] for m in all_metrics) / len(all_metrics),
            "avg_fp": sum(m["fp"] for m in all_metrics) / len(all_metrics),
            "avg_fn": sum(m["fn"] for m in all_metrics) / len(all_metrics)
        }
        
        return {
            "metrics": avg_metrics,
            "per_example_metrics": all_metrics,
            "predictions": predictions,
            "n_examples": len(dataset)
        }


class LEDGAREvaluator:
    """Evaluator for LEDGAR clause classification task."""
    
    def __init__(self, agent: BaseLegalAgent):
        """
        Initialize LEDGAR evaluator.
        
        Args:
            agent: Agent implementing classify_clause() method
        """
        self.agent = agent
    
    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate agent on LEDGAR dataset.
        
        Args:
            dataset: List of examples with format:
                {
                    "input": {"clause_text": str},
                    "gold": str (gold category label),
                    "metadata": {...}
                }
        
        Returns:
            Dictionary with metrics and predictions
        """
        predictions = []
        predicted_labels = []
        gold_labels = []
        
        for example in tqdm(dataset, desc="Evaluating LEDGAR"):
            input_data = example["input"]
            clause_text = input_data["clause_text"]
            gold_label = example["gold"]
            
            # Get prediction
            pred_label = self.agent.classify_clause(clause_text)
            
            predictions.append({
                "clause_text": clause_text[:200],  # Store first 200 chars for reference
                "predicted_label": pred_label,
                "gold_label": gold_label
            })
            
            predicted_labels.append(pred_label)
            gold_labels.append(gold_label)
        
        # Compute macro F1 and per-class metrics
        macro_metrics = compute_macro_f1(predicted_labels, gold_labels)
        
        # Note: Calibration metrics require probability predictions
        # For now, we skip calibration if agent doesn't provide probabilities
        calibration_metrics = compute_calibration_metrics(
            predicted_labels, 
            gold_labels, 
            predicted_probs=None
        )
        
        return {
            "metrics": {
                "macro_f1": macro_metrics["macro_f1"],
                "weighted_f1": macro_metrics["weighted_f1"],
                "micro_f1": macro_metrics["micro_f1"],
                "calibration_error": calibration_metrics.get("calibration_error"),
                "brier_score": calibration_metrics.get("brier_score")
            },
            "per_class_metrics": macro_metrics["per_class_f1"],
            "confusion_matrix": macro_metrics["confusion_matrix"],
            "predictions": predictions,
            "n_examples": len(dataset)
        }

