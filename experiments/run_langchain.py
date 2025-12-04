# Basic LangChain experiment runner

import os
import sys

# Get project root directory and add to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from orchestrators.langchain_agent import LangChainLegalAgent
from rag.indexer import FaissIndexer

def main():
    # Load the built ECHR index
    print("Loading ECHR index...")
    index_path = os.path.join(PROJECT_ROOT, "indexes", "echr")
    indexer = FaissIndexer(index_path=index_path)
    
    # Create the LangChain agent with Groq (fast inference)
    print("Initializing LangChain agent with Groq...")
    agent = LangChainLegalAgent(
        indexer, 
        model_name="llama-3.3-70b-versatile",  # Groq model
        k=3
    )
    
    # Test with sample queries
    print("\n" + "="*60)
    print("Testing LangChain RAG Agent")
    print("="*60 + "\n")
    
    test_queries = [
        "What are the key legal principles in human rights cases?",
        "What is the standard for determining violations of human rights?",
        "How do courts assess proportionality in human rights cases?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 60)
        try:
            answer = agent.ask(query)
            print(f"Answer: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("="*60)
    print("Testing complete!")

if __name__ == "__main__":
    main()

