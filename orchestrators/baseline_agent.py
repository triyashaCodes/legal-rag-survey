# Baseline agent: Simple retrieval + single LLM call (no orchestration)

from typing import List, Optional
from rag.indexer import FaissIndexer
from orchestrators.base_agent import BaseLegalAgent

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    raise ImportError(
        "langchain-groq and langchain-core are required for baseline agent. "
        "Install with: pip install langchain-groq langchain-core"
    )


class BaselineAgent(BaseLegalAgent):
    """
    Baseline agent with no orchestration features.
    
    This agent performs:
    - Simple retrieval (single pass, fixed k)
    - Single LLM call (no multi-step reasoning)
    - No tool calling, no planning, no iterative refinement
    
    This serves as the comparison baseline to measure the value
    of orchestration frameworks.
    """
    
    def __init__(
        self,
        indexer: FaissIndexer,
        model_name: str = "llama-3.3-70b-versatile",
        k: int = 3,
        temperature: float = 0
    ):
        """
        Initialize baseline agent.
        
        Args:
            indexer: FaissIndexer instance with loaded index
            model_name: Groq model name
            k: Number of documents to retrieve
            temperature: Temperature for LLM generation
        """
        super().__init__(indexer)
        self.k = k
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
        )
    
    def extract_spans(self, question: str, context: Optional[str] = None) -> str:
        """
        Extract answer spans using simple retrieval + single LLM call.
        
        Args:
            question: The question about a contract clause
            context: Optional context (if provided, use it; otherwise retrieve)
            
        Returns:
            Extracted answer span(s) as a string
        """
        # Retrieve documents if context not provided
        if context is None:
            docs = self.indexer.search(question, k=self.k)
            context = "\n\n".join(docs)
        
        # Single prompt for extraction
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Extract the exact text span(s) from the "
                      "provided contract context that answer the question. Return only the "
                      "relevant text span(s), nothing else."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nExtracted span(s):")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        return response.content.strip()
    
    def classify_clause(self, clause_text: str) -> str:
        """
        Classify clause using simple retrieval + single LLM call.
        
        Args:
            clause_text: The text of the legal clause to classify
            
        Returns:
            Predicted category label as a string
        """
        # Retrieve similar clauses for context
        docs = self.indexer.search(clause_text, k=self.k)
        context = "\n\n".join(docs)
        
        # Single prompt for classification
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Classify the given clause into one of the "
                      "LEDGAR categories. Return only the category label, nothing else."),
            ("human", "Similar clauses for reference:\n{context}\n\n"
                     "Clause to classify:\n{clause_text}\n\nCategory:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "clause_text": clause_text
        })
        
        return response.content.strip()
    
    def predict_violations(self, case_text: str) -> List[str]:
        """
        Predict violations using simple retrieval + single LLM call.
        
        Note: This baseline does NOT handle long context well - it just retrieves
        similar cases and uses a single LLM call. This is intentionally simple
        to show the value of orchestration for long-context tasks.
        
        Args:
            case_text: The full text of the ECHR case
            
        Returns:
            List of violated article identifiers
        """
        # For baseline, we retrieve similar cases (not the full case text)
        # This simulates a simple RAG approach without long-context handling
        query = case_text[:500]  # Use first 500 chars as query
        docs = self.indexer.search(query, k=self.k)
        context = "\n\n".join(docs)
        
        # Single prompt for violation prediction
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Analyze the case and predict which ECHR "
                      "articles were violated. Return a comma-separated list of article "
                      "identifiers (e.g., 'Article 6, Article 8'). If no violations, return 'None'."),
            ("human", "Similar cases for reference:\n{context}\n\n"
                     "Case to analyze (first 2000 chars):\n{case_text}\n\nViolated articles:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "case_text": case_text[:2000]  # Truncate for baseline
        })
        
        # Parse response into list
        result = response.content.strip()
        if result.lower() == "none" or not result:
            return []
        
        # Split by comma and clean up
        articles = [a.strip() for a in result.split(",")]
        return articles
    
    def ask(self, query: str) -> str:
        """
        Generic Q&A method for backward compatibility.
        
        Args:
            query: The question to answer
            
        Returns:
            Answer string
        """
        docs = self.indexer.search(query, k=self.k)
        context = "\n\n".join(docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Answer questions based on the provided "
                      "legal documents. If the context doesn't contain enough information, say so."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        return response.content

