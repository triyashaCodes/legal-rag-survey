# LangGraph agent orchestrator

from langgraph.graph import StateGraph
from rag.indexer import FaissIndexer
from langchain.chat_models import ChatOpenAI

class LangGraphLegalAgent:
    def __init__(self, indexer: FaissIndexer):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.indexer = indexer
        graph = StateGraph()

        @graph.add_node
        def retrieve(state):
            state["docs"] = self.indexer.search(state["query"], k=3)
            return state

        @graph.add_node
        def answer(state):
            prompt = f"Context:\n{state['docs']}\n\nQuestion:\n{state['query']}"
            response = self.llm.invoke(prompt)
            state["answer"] = response.content
            return state

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer")

        self.app = graph.compile()

    def ask(self, query):
        output = self.app.invoke({"query": query})
        return output["answer"]
