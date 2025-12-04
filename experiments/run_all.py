# Run complete evaluation suite across all frameworks and tasks

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.run_evaluation import run_all_tasks
from utils.config import DATASET_PATHS, get_config, ensure_directories
from orchestrators.baseline_agent import BaselineAgent
from orchestrators.langchain_agent import LangChainLegalAgent
from orchestrators.langgraph_agent import LangGraphLegalAgent

# Ensure directories exist
ensure_directories()

# Framework factories
def create_baseline(indexer):
    return BaselineAgent(indexer, **get_config("baseline"))

def create_langchain(indexer):
    return LangChainLegalAgent(indexer, **get_config("langchain"))

def create_langgraph(indexer):
    return LangGraphLegalAgent(indexer, **get_config("langgraph"))

# All frameworks to evaluate
frameworks = {
    "baseline": create_baseline,
    "langchain": create_langchain,
    "langgraph": create_langgraph
}

print(f"\n{'='*60}")
print("Running Complete Evaluation Suite")
print(f"{'='*60}\n")

all_results = {}

for framework_name, factory in frameworks.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {framework_name.upper()} on all tasks")
    print(f"{'='*60}\n")
    
    results = run_all_tasks(
        framework_name=framework_name,
        agent_factory=factory,
        dataset_paths=DATASET_PATHS,
        sample_size=50  # Limit for testing - remove for full evaluation
    )
    
    all_results[framework_name] = results

print(f"\n{'='*60}")
print("Complete evaluation suite finished!")
print(f"{'='*60}\n")

# Print summary
print("Summary:")
for framework_name, results in all_results.items():
    print(f"\n{framework_name.upper()}:")
    for task, task_results in results.items():
        metrics = task_results.get("metrics", {})
        if "f1" in metrics:
            print(f"  {task}: F1 = {metrics['f1']:.4f}")
        elif "macro_f1" in metrics:
            print(f"  {task}: Macro F1 = {metrics['macro_f1']:.4f}")
        elif "exact_match_f1" in metrics:
            print(f"  {task}: Exact Match F1 = {metrics['exact_match_f1']:.4f}")

