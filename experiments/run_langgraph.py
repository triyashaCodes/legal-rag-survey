# LangGraph experiment runner

from orchestrators.langgraph_agent import LangGraphLegalAgent
from rag.indexer import FaissIndexer
from eval.evaluator import Evaluator
from eval.dataset_wrappers import load_CUAD

indexer = FaissIndexer()
indexer.add_documents([
    "This contract limits liability to $100,000.",
    "The buyer must provide a 30-day notice.",
    "Indemnification applies to third-party claims."
])

agent = LangGraphLegalAgent(indexer)
evaluator = Evaluator(agent)

dataset = load_CUAD("data/CUAD/sample.json")

preds, gold = evaluator.evaluate(dataset)

print(preds[:5])
