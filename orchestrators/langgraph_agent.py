# LangGraph agent orchestrator with task-specific graph workflows

from typing import List, Optional, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from rag.indexer import FaissIndexer
from orchestrators.base_agent import BaseLegalAgent

try:
    from langchain_groq import ChatGroq
except ImportError:
    raise ImportError(
        "langchain-groq is not installed. Install with: pip install langchain-groq"
    )


class LangGraphLegalAgent(BaseLegalAgent):
    """
    LangGraph agent with state-machine orchestration for complex workflows.
    
    Uses graph nodes for:
    - CUAD: retrieve → extract → verify
    - ECHR: iterative graph with long-context routing
    - LEDGAR: parallel classification nodes
    """
    
    def __init__(
        self, 
        indexer: FaissIndexer, 
        model_name: str = "llama-3.3-70b-versatile",
        k: int = 3,
        temperature: float = 0
    ):
        """
        Initialize LangGraph agent.
        
        Args:
            indexer: FaissIndexer instance with loaded index
            model_name: Groq model name
            k: Number of documents to retrieve
            temperature: Temperature for LLM generation
        """
        super().__init__(indexer)
        self.k = k
        self.llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=None)
    
    def extract_spans(self, question: str, context: Optional[str] = None) -> str:
        """
        Extract spans using graph: retrieve → extract → verify.
        
        Args:
            question: The question about a contract clause
            context: Optional context (if provided, use it; otherwise retrieve)
            
        Returns:
            Extracted answer span(s) as a string
        """
        # Define state for CUAD extraction
        class ExtractionState(TypedDict):
            question: str
            context: str
            docs: List[str]
            extracted_spans: str
            verified: bool
        
        graph = StateGraph(ExtractionState)
        
        def retrieve_node(state: ExtractionState) -> ExtractionState:
            """Retrieve relevant documents"""
            if context:
                state["docs"] = [context]
            else:
                state["docs"] = self.indexer.search(state["question"], k=self.k)
            state["context"] = "\n\n".join(state["docs"])
            return state
        
        def extract_node(state: ExtractionState) -> ExtractionState:
            """Extract spans from context"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a legal contract analyst. Extract the exact text span(s) "
                          "from the provided contract context that answer the question. "
                          "Return only the relevant text span(s), nothing else."),
                ("human", "Contract Context:\n{context}\n\nQuestion: {question}\n\nExtracted span(s):")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": state["context"],
                "question": state["question"]
            })
            state["extracted_spans"] = response.content.strip()
            return state
        
        def verify_node(state: ExtractionState) -> ExtractionState:
            """Verify extraction quality (simple check)"""
            # Simple verification: check if span is non-empty
            state["verified"] = bool(state["extracted_spans"])
            return state
        
        # Build graph
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("extract", extract_node)
        graph.add_node("verify", verify_node)
        
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "extract")
        graph.add_edge("extract", "verify")
        graph.add_edge("verify", END)
        
        app = graph.compile()
        
        # Run graph
        result = app.invoke({
            "question": question,
            "context": "",
            "docs": [],
            "extracted_spans": "",
            "verified": False
        })
        
        return result["extracted_spans"]
    
    def classify_clause(self, clause_text: str) -> str:
        """
        Classify clause using parallel classification nodes.
        
        Args:
            clause_text: The text of the legal clause to classify
            
        Returns:
            Predicted category label as a string
        """
        # Define state for LEDGAR classification
        class ClassificationState(TypedDict):
            clause_text: str
            context: str
            docs: List[str]
            predicted_label: str
        
        graph = StateGraph(ClassificationState)
        
        def retrieve_node(state: ClassificationState) -> ClassificationState:
            """Retrieve similar clauses"""
            state["docs"] = self.indexer.search(state["clause_text"], k=self.k)
            state["context"] = "\n\n".join(state["docs"])
            return state
        
        def classify_node(state: ClassificationState) -> ClassificationState:
            """Classify the clause"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a legal assistant. Classify the given clause into one of the "
                          "LEDGAR categories. Return only the category label, nothing else."),
                ("human", "Similar clauses for reference:\n{context}\n\n"
                         "Clause to classify:\n{clause_text}\n\nCategory:")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": state["context"],
                "clause_text": state["clause_text"]
            })
            state["predicted_label"] = response.content.strip()
            return state
        
        # Build graph
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("classify", classify_node)
        
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "classify")
        graph.add_edge("classify", END)
        
        app = graph.compile()
        
        # Run graph
        result = app.invoke({
            "clause_text": clause_text,
            "context": "",
            "docs": [],
            "predicted_label": ""
        })
        
        return result["predicted_label"]
    
    def predict_violations(self, case_text: str) -> List[str]:
        """
        Predict violations using iterative graph with long-context routing.
        
        Args:
            case_text: The full text of the ECHR case
            
        Returns:
            List of violated article identifiers
        """
        # Define state for ECHR violation prediction
        class ViolationState(TypedDict):
            case_text: str
            key_facts: str
            context: str
            docs: List[str]
            predicted_violations: List[str]
            iteration: int
            max_iterations: int
        
        graph = StateGraph(ViolationState)
        
        def extract_facts_node(state: ViolationState) -> ViolationState:
            """Extract key facts from case (first iteration)"""
            if state["iteration"] == 0:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a legal analyst. Extract key facts and legal issues from the case."),
                    ("human", "Case text (first 3000 chars):\n{case_text}\n\nKey facts and issues:")
                ])
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "case_text": state["case_text"][:3000]
                })
                state["key_facts"] = response.content
            return state
        
        def retrieve_node(state: ViolationState) -> ViolationState:
            """Retrieve similar cases"""
            query = state["key_facts"][:500] if state["key_facts"] else state["case_text"][:500]
            state["docs"] = self.indexer.search(query, k=self.k)
            state["context"] = "\n\n".join(state["docs"])
            return state
        
        def predict_node(state: ViolationState) -> ViolationState:
            """Predict violations"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a legal assistant specializing in ECHR cases. "
                          "Analyze the case facts and similar cases to predict which ECHR articles were violated. "
                          "Return a comma-separated list of article identifiers (e.g., 'Article 6, Article 8'). "
                          "If no violations, return 'None'."),
                ("human", "Similar cases:\n{context}\n\n"
                         "Key facts from case:\n{key_facts}\n\n"
                         "Full case text (first 5000 chars):\n{case_text}\n\nViolated articles:")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": state["context"],
                "key_facts": state["key_facts"],
                "case_text": state["case_text"][:5000]
            })
            
            result = response.content.strip()
            if result.lower() == "none" or not result:
                state["predicted_violations"] = []
            else:
                state["predicted_violations"] = [a.strip() for a in result.split(",")]
            
            state["iteration"] += 1
            return state
        
        def should_continue(state: ViolationState) -> str:
            """Decide whether to continue iterating"""
            if state["iteration"] >= state["max_iterations"]:
                return "end"
            # Simple check: if we have predictions, we're done
            if state["predicted_violations"]:
                return "end"
            return "continue"
        
        # Build graph with conditional routing
        graph.add_node("extract_facts", extract_facts_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("predict", predict_node)
        
        graph.set_entry_point("extract_facts")
        graph.add_edge("extract_facts", "retrieve")
        graph.add_edge("retrieve", "predict")
        graph.add_conditional_edges("predict", should_continue, {
            "end": END,
            "continue": "retrieve"  # Iterate by retrieving again
        })
        
        app = graph.compile()
        
        # Run graph
        result = app.invoke({
            "case_text": case_text,
            "key_facts": "",
            "context": "",
            "docs": [],
            "predicted_violations": [],
            "iteration": 0,
            "max_iterations": 2  # Max 2 iterations
        })
        
        return result["predicted_violations"]
    
    def ask(self, query: str) -> str:
        """
        Generic Q&A method for backward compatibility.
        
        Args:
            query: The question to answer
            
        Returns:
            Answer string
        """
        class QAState(TypedDict):
            query: str
            docs: List[str]
            answer: str
        
        graph = StateGraph(QAState)
        
        def retrieve(state: QAState) -> QAState:
            state["docs"] = self.indexer.search(state["query"], k=self.k)
            return state
        
        def answer(state: QAState) -> QAState:
            context = "\n\n".join(state["docs"])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a legal assistant. Answer questions based on the provided legal documents."),
                ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "query": state["query"]
            })
            state["answer"] = response.content
            return state
        
        graph.add_node("retrieve", retrieve)
        graph.add_node("answer", answer)
        
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        
        app = graph.compile()
        output = app.invoke({"query": query, "docs": [], "answer": ""})
        return output["answer"]
