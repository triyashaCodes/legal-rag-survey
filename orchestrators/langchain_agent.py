# Basic LangChain RAG orchestrator with Groq

from langchain.prompts import ChatPromptTemplate
from rag.indexer import FaissIndexer
from typing import List, Optional

try:
    from langchain_groq import ChatGroq
except ImportError:
    raise ImportError(
        "langchain-groq is not installed. Install with: pip install langchain-groq"
    )


class LangChainLegalAgent:
    """Basic LangChain RAG agent for legal document Q&A using Groq"""
    
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
        self.indexer = indexer
        self.k = k
        
        # Initialize Groq LLM
        model_name = model_name or "llama-3.3-70b-versatile"
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant. Answer questions based on the provided legal documents. "
                      "If the context doesn't contain enough information, say so."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
    
    def _retrieve_documents(self, query: str) -> List[str]:
        """Retrieve relevant documents from the index"""
        return self.indexer.search(query, k=self.k)
    
    def ask(self, query: str) -> str:
        """
        Answer a question using RAG
        
        Args:
            query: The question to answer
            
        Returns:
            The answer string
        """
        # Retrieve relevant documents
        documents = self._retrieve_documents(query)
        
        # Combine documents into context
        context = "\n\n".join(documents)
        
        # Create chain and generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        return response.content

