# Unified evaluation pipeline

import os
import json
from typing import Dict, Any, Optional
from orchestrators.base_agent import BaseLegalAgent
from eval.dataset_wrappers import load_CUAD, load_ECHR, load_LEDGAR
from eval.task_evaluators import CUADEvaluator, ECHREvaluator, LEDGAREvaluator


def run_evaluation(
    framework_name: str,
    agent: BaseLegalAgent,
    task: str,
    dataset_path: str,
    output_dir: Optional[str] = None,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run evaluation for a framework on a specific task.
    
    Args:
        framework_name: Name of the framework (e.g., "baseline", "langchain")
        agent: Agent instance implementing BaseLegalAgent
        task: Task name ("CUAD", "ECHR", or "LEDGAR")
        dataset_path: Path to evaluation dataset JSON file
        output_dir: Directory to save results (default: results/{framework}/{task}/)
        sample_size: Optional limit on number of examples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    print(f"Loading {task} dataset from {dataset_path}...")
    if task == "CUAD":
        dataset = load_CUAD(dataset_path)
    elif task == "ECHR":
        dataset = load_ECHR(dataset_path)
    elif task == "LEDGAR":
        dataset = load_LEDGAR(dataset_path)
    else:
        raise ValueError(f"Unknown task: {task}. Must be one of: CUAD, ECHR, LEDGAR")
    
    # Sample if requested
    if sample_size and sample_size < len(dataset):
        print(f"Sampling {sample_size} examples from {len(dataset)} total...")
        dataset = dataset[:sample_size]
    
    # Run evaluation
    print(f"Evaluating {framework_name} on {task} ({len(dataset)} examples)...")
    if task == "CUAD":
        evaluator = CUADEvaluator(agent)
    elif task == "ECHR":
        evaluator = ECHREvaluator(agent)
    elif task == "LEDGAR":
        evaluator = LEDGAREvaluator(agent)
    
    results = evaluator.evaluate(dataset)
    
    # Add metadata
    results["metadata"] = {
        "framework": framework_name,
        "task": task,
        "dataset_path": dataset_path,
        "n_examples": len(dataset),
        "sample_size": sample_size
    }
    
    # Save results
    if output_dir is None:
        output_dir = os.path.join("results", framework_name, task)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")
    
    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation complete! Metrics:")
    for metric_name, metric_value in results["metrics"].items():
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
        elif isinstance(metric_value, dict):
            print(f"  {metric_name}:")
            for k, v in metric_value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
    
    return results


def run_all_tasks(
    framework_name: str,
    agent_factory,
    dataset_paths: Dict[str, str],
    output_base_dir: str = "results",
    sample_size: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run evaluation for a framework on all three tasks.
    
    Args:
        framework_name: Name of the framework
        agent_factory: Function that creates an agent instance (takes indexer as arg)
        dataset_paths: Dictionary mapping task names to dataset paths
            e.g., {"CUAD": "data/CUAD/cuad_eval.json", ...}
        output_base_dir: Base directory for results
        sample_size: Optional limit on number of examples per task
        
    Returns:
        Dictionary mapping task names to results
    """
    from rag.indexer import FaissIndexer
    
    all_results = {}
    
    # Index paths for each task
    index_paths = {
        "CUAD": os.path.join("indexes", "cuad"),
        "ECHR": os.path.join("indexes", "echr"),
        "LEDGAR": os.path.join("indexes", "ledgar")
    }
    
    for task in ["CUAD", "ECHR", "LEDGAR"]:
        if task not in dataset_paths:
            print(f"Skipping {task} - no dataset path provided")
            continue
        
        # Load appropriate index
        index_path = index_paths.get(task)
        if not index_path or not os.path.exists(index_path + ".index"):
            print(f"Warning: Index not found for {task} at {index_path}")
            print(f"  Skipping {task} evaluation")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {framework_name} on {task}")
        print(f"{'='*60}\n")
        
        indexer = FaissIndexer(index_path=index_path)
        agent = agent_factory(indexer)
        
        results = run_evaluation(
            framework_name=framework_name,
            agent=agent,
            task=task,
            dataset_path=dataset_paths[task],
            output_dir=os.path.join(output_base_dir, framework_name, task),
            sample_size=sample_size
        )
        
        all_results[task] = results
    
    return all_results

