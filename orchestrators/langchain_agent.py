# LangChain RAG orchestrator with Groq - Enhanced with task-specific methods

from langchain_core.prompts import ChatPromptTemplate
from rag.indexer import FaissIndexer
from typing import List, Optional
from orchestrators.base_agent import BaseLegalAgent

try:
    from langchain_groq import ChatGroq
except ImportError:
    raise ImportError(
        "langchain-groq is not installed. Install with: pip install langchain-groq"
    )


class LangChainLegalAgent(BaseLegalAgent):
    """LangChain RAG agent with task-specific orchestration methods"""
    
    def __init__(
        self, 
        indexer: FaissIndexer, 
        model_name: Optional[str] = None,
        k: int = 3,
        temperature: float = 0
    ):
        """
        Initialize the LangChain agent with Groq
        
        Args:
            indexer: FaissIndexer instance with loaded index
            model_name: Groq model name (defaults to "llama-3.3-70b-versatile")
            k: Number of documents to retrieve
            temperature: Temperature for LLM generation
        """
        super().__init__(indexer)
        self.k = k
        
        # Initialize Groq LLM
        model_name = model_name or "llama-3.3-70b-versatile"
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
        )
        
        # Create base prompt template for Q&A
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Answer questions based on the provided legal documents. "
                      "If the context doesn't contain enough information, say so."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
    
    def _retrieve_documents(self, query: str) -> List[str]:
        """Retrieve relevant documents from the index"""
        return self.indexer.search(query, k=self.k)
    
    def extract_spans(self, question: str, context: Optional[str] = None) -> str:
        """
        Extract answer spans using LangChain with retrieval loop.
        
        For CUAD task: Uses tool-calling for iterative retrieval and extraction.
        
        Args:
            question: The question about a contract clause
            context: Optional context (if provided, use it; otherwise retrieve)
            
        Returns:
            Extracted answer span(s) as a string
        """
        # If context provided, use it directly
        if context:
            docs = [context]
        else:
            # Retrieve with iterative refinement: first broad, then focused
            docs = self._retrieve_documents(question)
            
            # If first retrieval doesn't seem sufficient, do a second pass
            if len(docs) < self.k:
                # Try a more specific query
                focused_query = f"{question} contract clause"
                additional_docs = self._retrieve_documents(focused_query)
                docs = list(dict.fromkeys(docs + additional_docs))[:self.k]
        
        context_text = "\n\n".join(docs)
        
        # Multi-step extraction prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal contract analyst. Extract the exact text span(s) from "
                      "the provided contract context that answer the question. "
                      "Return only the relevant text span(s), nothing else. "
                      "If multiple spans are needed, return them separated by newlines."),
            ("human", "Contract Context:\n{context}\n\nQuestion: {question}\n\nExtracted span(s):")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return response.content.strip()
    
    def classify_clause(self, clause_text: str) -> str:
        """
        Classify clause using LangChain with batch processing capability.
        
        For LEDGAR task: Uses LCEL for efficient classification.
        
        Args:
            clause_text: The text of the legal clause to classify
            
        Returns:
            Predicted category label as a string
        """
        # Retrieve similar clauses for context
        docs = self._retrieve_documents(clause_text)
        context = "\n\n".join(docs)
        
        # Classification prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Classify the given clause into one of the "
                      "LEDGAR categories based on the clause content and similar examples. "
                      "Return only the category label, nothing else."),
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
        Predict violations using LangChain with multi-step reasoning.
        
        For ECHR task: Uses iterative retrieval and multi-step chain for long-context reasoning.
        
        Args:
            case_text: The full text of the ECHR case
            
        Returns:
            List of violated article identifiers
        """
        # Multi-step approach: chunk the case and analyze iteratively
        # Step 1: Extract key facts
        facts_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal analyst. Extract key facts and legal issues from the case."),
            ("human", "Case text (first 3000 chars):\n{case_text}\n\nKey facts and issues:")
        ])
        
        facts_chain = facts_prompt | self.llm
        facts_response = facts_chain.invoke({
            "case_text": case_text[:3000]
        })
        key_facts = facts_response.content
        
        # Step 2: Retrieve similar cases
        query = key_facts[:500] if len(key_facts) > 500 else key_facts
        docs = self._retrieve_documents(query)
        context = "\n\n".join(docs)
        
        # Step 3: Predict violations based on facts and similar cases
        violation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant specializing in ECHR cases. "
                      "Analyze the case facts and similar cases to predict which ECHR articles were violated. "
                      "Return a comma-separated list of article identifiers (e.g., 'Article 6, Article 8'). "
                      "If no violations, return 'None'."),
            ("human", "Similar cases:\n{context}\n\n"
                     "Key facts from case:\n{key_facts}\n\n"
                     "Full case text (first 5000 chars):\n{case_text}\n\nViolated articles:")
        ])
        
        violation_chain = violation_prompt | self.llm
        response = violation_chain.invoke({
            "context": context,
            "key_facts": key_facts,
            "case_text": case_text[:5000]
        })
        
        # Parse response
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
        documents = self._retrieve_documents(query)
        context = "\n\n".join(documents)
        
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        return response.content

