# LangGraph experiment runner

import os
import sys

# Get project root directory and add to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from orchestrators.langgraph_agent import LangGraphLegalAgent
from rag.indexer import FaissIndexer
from eval.evaluator import Evaluator
from eval.dataset_wrappers import load_CUAD

# Load the built ECHR index
print("Loading ECHR index...")
index_path = os.path.join(PROJECT_ROOT, "indexes", "echr")
indexer = FaissIndexer(index_path=index_path)

# Create the LangGraph agent
print("Initializing LangGraph agent...")
agent = LangGraphLegalAgent(indexer)

# Example: Test with a simple query
print("\nTesting with a sample query...")
test_query = "What are the key legal principles in human rights cases?"
answer = agent.ask(test_query)
print(f"Query: {test_query}")
print(f"Answer: {answer}\n")

# If you have evaluation data, uncomment below:
# evaluator = Evaluator(agent)
# dataset = load_CUAD("data/CUAD/sample.json")
# preds, gold = evaluator.evaluate(dataset)
# print(f"Evaluated {len(preds)} examples")
