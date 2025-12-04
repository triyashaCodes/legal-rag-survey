# Quick test script to verify agents work without full evaluation

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag.indexer import FaissIndexer
from orchestrators.baseline_agent import BaselineAgent
from orchestrators.langchain_agent import LangChainLegalAgent
from orchestrators.langgraph_agent import LangGraphLegalAgent


def test_agent(agent, agent_name):
    """Test an agent with sample queries for each task."""
    print(f"\n{'='*60}")
    print(f"Testing {agent_name.upper()}")
    print(f"{'='*60}\n")
    
    # Test CUAD: Extract spans
    print("1. CUAD Task - Span Extraction")
    print("-" * 60)
    cuad_question = "What is the termination clause?"
    try:
        result = agent.extract_spans(cuad_question)
        print(f"Question: {cuad_question}")
        print(f"Extracted span: {result[:200]}..." if len(result) > 200 else f"Extracted span: {result}")
        print("✓ CUAD test passed\n")
    except Exception as e:
        print(f"✗ CUAD test failed: {e}\n")
    
    # Test LEDGAR: Classify clause
    print("2. LEDGAR Task - Clause Classification")
    print("-" * 60)
    clause_text = "The parties agree to maintain confidentiality of all proprietary information."
    try:
        result = agent.classify_clause(clause_text)
        print(f"Clause: {clause_text[:100]}...")
        print(f"Predicted category: {result}")
        print("✓ LEDGAR test passed\n")
    except Exception as e:
        print(f"✗ LEDGAR test failed: {e}\n")
    
    # Test ECHR: Predict violations
    print("3. ECHR Task - Violation Prediction")
    print("-" * 60)
    case_text = "The applicant was denied access to legal representation during interrogation."
    try:
        result = agent.predict_violations(case_text)
        print(f"Case: {case_text[:100]}...")
        print(f"Predicted violations: {result}")
        print("✓ ECHR test passed\n")
    except Exception as e:
        print(f"✗ ECHR test failed: {e}\n")
    
    # Test generic Q&A
    print("4. Generic Q&A")
    print("-" * 60)
    query = "What are the key legal principles?"
    try:
        result = agent.ask(query)
        print(f"Query: {query}")
        print(f"Answer: {result[:200]}..." if len(result) > 200 else f"Answer: {result}")
        print("✓ Q&A test passed\n")
    except Exception as e:
        print(f"✗ Q&A test failed: {e}\n")


def main():
    """Main test function."""
    print("="*60)
    print("Quick Agent Test Script")
    print("="*60)
    print("\nThis script tests the baseline, LangChain, and LangGraph agents")
    print("with sample queries for each task type.\n")
    
    # Check which indexes are available
    available_indexes = []
    index_paths = {
        "ECHR": os.path.join(PROJECT_ROOT, "indexes", "echr"),
        "LEDGAR": os.path.join(PROJECT_ROOT, "indexes", "ledgar"),
        "CUAD": os.path.join(PROJECT_ROOT, "indexes", "cuad")
    }
    
    for name, path in index_paths.items():
        if os.path.exists(path + ".index"):
            available_indexes.append((name, path))
            print(f"✓ Found {name} index")
        else:
            print(f"✗ Missing {name} index at {path}")
    
    if not available_indexes:
        print("\nError: No indexes found!")
        print("Please run: python utils/build_indexes.py")
        return
    
    # Use first available index for testing
    index_name, index_path = available_indexes[0]
    print(f"\nUsing {index_name} index for testing...\n")
    
    try:
        indexer = FaissIndexer(index_path=index_path)
        print(f"Loaded index with {len(indexer.docs)} documents\n")
    except Exception as e:
        print(f"Error loading index: {e}")
        return
    
    # Test Baseline Agent
    try:
        baseline = BaselineAgent(indexer, k=3)
        test_agent(baseline, "Baseline Agent")
    except Exception as e:
        print(f"Error creating Baseline agent: {e}\n")
    
    # Test LangChain Agent
    try:
        langchain = LangChainLegalAgent(indexer, k=3)
        test_agent(langchain, "LangChain Agent")
    except Exception as e:
        print(f"Error creating LangChain agent: {e}\n")
    
    # Test LangGraph Agent
    try:
        langgraph = LangGraphLegalAgent(indexer, k=3)
        test_agent(langgraph, "LangGraph Agent")
    except Exception as e:
        print(f"Error creating LangGraph agent: {e}\n")
    
    print("="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

