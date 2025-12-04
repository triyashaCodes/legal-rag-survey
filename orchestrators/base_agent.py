# Base agent interface for legal RAG orchestrators

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from rag.indexer import FaissIndexer


class BaseLegalAgent(ABC):
    """
    Abstract base class for legal RAG orchestrators.
    
    All framework implementations should inherit from this class and implement
    the three task-specific methods:
    - extract_spans: For CUAD (clause span extraction)
    - classify_clause: For LEDGAR (clause classification)
    - predict_violations: For ECHR (violation prediction)
    """
    
    def __init__(self, indexer: FaissIndexer, **kwargs):
        """
        Initialize the agent with a FAISS indexer.
        
        Args:
            indexer: FaissIndexer instance with loaded index
            **kwargs: Framework-specific initialization parameters
        """
        self.indexer = indexer
    
    @abstractmethod
    def extract_spans(self, question: str, context: Optional[str] = None) -> str:
        """
        Extract answer spans from legal documents (CUAD task).
        
        This method should:
        1. Retrieve relevant document chunks using the question
        2. Extract the answer span(s) that answer the question
        3. Return the extracted text span(s)
        
        Args:
            question: The question about a contract clause
            context: Optional context document (if provided, use it directly)
            
        Returns:
            Extracted answer span(s) as a string
        """
        pass
    
    @abstractmethod
    def classify_clause(self, clause_text: str) -> str:
        """
        Classify a legal clause into a category (LEDGAR task).
        
        This method should:
        1. Classify the given clause text into one of the LEDGAR categories
        2. Return the predicted category label
        
        Args:
            clause_text: The text of the legal clause to classify
            
        Returns:
            Predicted category label as a string
        """
        pass
    
    @abstractmethod
    def predict_violations(self, case_text: str) -> List[str]:
        """
        Predict which human rights articles were violated (ECHR task).
        
        This method should:
        1. Analyze the case text (potentially long context)
        2. Predict which ECHR articles were violated
        3. Return a list of violated article identifiers
        
        Args:
            case_text: The full text of the ECHR case
            
        Returns:
            List of violated article identifiers (e.g., ["Article 6", "Article 8"])
        """
        pass
    
    def ask(self, query: str) -> str:
        """
        Generic Q&A method for backward compatibility.
        
        This is a convenience method that can be overridden by frameworks
        that want to provide a unified interface. By default, it's a simple
        retrieval + answer generation.
        
        Args:
            query: The question to answer
            
        Returns:
            Answer string
        """
        # Default implementation: simple retrieval + answer
        docs = self.indexer.search(query, k=3)
        context = "\n\n".join(docs)
        return f"Context: {context}\n\nQuestion: {query}"

