# Run all frameworks on ECHR task

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag.indexer import FaissIndexer
from eval.run_evaluation import run_evaluation
from utils.config import DATASET_PATHS, INDEX_PATHS, get_config, ensure_directories
from orchestrators.baseline_agent import BaselineAgent
from orchestrators.langchain_agent import LangChainLegalAgent
from orchestrators.langgraph_agent import LangGraphLegalAgent

# Ensure directories exist
ensure_directories()

# Load ECHR index
print("Loading ECHR index...")
index_path = INDEX_PATHS["ECHR"]
if not os.path.exists(index_path + ".index"):
    print(f"Error: Index not found at {index_path}")
    print("Please run: python utils/build_indexes.py")
    sys.exit(1)

indexer = FaissIndexer(index_path=index_path)

# Framework factories
def create_baseline(indexer):
    config = get_config("baseline", "ECHR")
    return BaselineAgent(indexer, **config)

def create_langchain(indexer):
    config = get_config("langchain", "ECHR")
    return LangChainLegalAgent(indexer, **config)

def create_langgraph(indexer):
    config = get_config("langgraph", "ECHR")
    return LangGraphLegalAgent(indexer, **config)

# Run evaluations
frameworks = {
    "baseline": create_baseline,
    "langchain": create_langchain,
    "langgraph": create_langgraph
}

dataset_path = DATASET_PATHS["ECHR"]

print(f"\n{'='*60}")
print("Running ECHR Evaluation")
print(f"{'='*60}\n")

for framework_name, factory in frameworks.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {framework_name}")
    print(f"{'='*60}\n")
    
    agent = factory(indexer)
    run_evaluation(
        framework_name=framework_name,
        agent=agent,
        task="ECHR",
        dataset_path=dataset_path,
        sample_size=50  # Limit for testing - remove for full evaluation
    )

print(f"\n{'='*60}")
print("ECHR evaluation complete!")
print(f"{'='*60}")

