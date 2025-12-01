
from typing import Annotated, Sequence, TypedDict, Literal,List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from evaluation.ragas import evaluate_context_precision, evaluate_response_relevancy

from tavily import TavilyClient
import os


class AgenticRAG:
    """Agentic RAG using clean tool-flag based routing."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        query: str
        current_query: str
        retrieved_docs: str | None
        retries: int
        tool: str | None   # 👈 THIS IS THE IMPORTANT PART

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helper ----------
    def _format_docs(self, docs):
        if not docs:
            return "No relevant documents found."

        formatted = []
        for d in docs:
            meta = d.metadata or {}
            formatted.append(
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
        return "\n\n---\n\n".join(formatted)

    # ---------- Nodes ----------

    def _assistant(self, state: AgentState):
        print("--- ASSISTANT ---")

        query = state["current_query"]

        # Decide which tool to use
        if any(word in query.lower() for word in ["price", "review", "product"]):
            return {
                "tool": "retriever",
                "messages": [HumanMessage(content="I will use vector retriever")]
            }

        # No tool -> answer directly
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer directly.\n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": query})

        return {
            "tool": None,
            "messages": [HumanMessage(content=response)]
        }

    def _retriever(self, state: AgentState):
        print("--- RETRIEVER TOOL ---")

        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(state["current_query"])
        context = self._format_docs(docs)

        return {
            "retrieved_docs": context,
            "tool": None,  # ✅ reset tool after use
            "messages": [HumanMessage(content="Retriever completed")]
        }

    def _grader(self, state: AgentState) -> Literal["generate", "rewrite", "tavily"]:
        print("--- GRADER ---")

        retries = state.get("retries", 0)

        if retries >= 2:
            return "tavily"

        prompt = PromptTemplate(
            template="""Query: {query}
Docs: {docs}
Are docs relevant? Answer yes or no.""",
            input_variables=["query", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke(
            {"query": state["current_query"], "docs": state["retrieved_docs"]}
        )

        return "generate" if "yes" in score.lower() else "rewrite"

    def _rewrite(self, state: AgentState):
        print("--- REWRITE TOOL ---")

        retries = state["retries"] + 1

        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query: {state['current_query']}")]
        )

        return {
            "current_query": new_q.content,
            "retries": retries,
            "tool": "retriever",  # try retriever again after rewrite
            "messages": [HumanMessage(content=f"Rewritten: {new_q.content}")]
        }

    def _tavily(self, state: AgentState):
        print("--- TAVILY TOOL ---")

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query=state["current_query"], max_results=3)

        content = "\n\n".join(
            [item.get("content", "") for item in result.get("results", [])]
        )

        return {
            "retrieved_docs": content,
            "tool": None,
            "messages": [HumanMessage(content="Used Tavily web search")]
        }

    def _generate(self, state: AgentState):
        print("--- GENERATOR ---")

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "context": state["retrieved_docs"],
                "question": state["current_query"],
            }
        )

        return {
            "tool": None,
            "messages": [HumanMessage(content=response)]
        }

    # ---------- Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)

        workflow.add_node("Assistant", self._assistant)
        workflow.add_node("Retriever", self._retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("Tavily", self._tavily)

        workflow.add_edge(START, "Assistant")

        # Tool-based routing
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if state.get("tool") == "retriever" else END,
            {"Retriever": "Retriever", END: END},
        )

        workflow.add_conditional_edges(
            "Retriever",
            self._grader,
            {
                "generate": "Generator",
                "rewrite": "Rewriter",
                "tavily": "Tavily",
            },
        )

        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        workflow.add_edge("Tavily", "Generator")

        return workflow

    # ---------- Runner ----------
    def run(self, query: str, thread_id: str = "default") -> str:
        result = self.app.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "current_query": query,
                "retrieved_docs": None,
                "retries": 0,
                "tool": None,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return {
        "answer": result["messages"][-1].content,
        "retrieved_docs": result.get("retrieved_docs")
        }


if __name__ == "__main__":
    
    
    rag_agent = AgenticRAG()
    user_query="What is the price of iPhone 16?"
    answer = rag_agent.run(user_query)
    response=answer["answer"]
    print("\nFinal Answer:\n", answer['answer'])
    print("\nRetrieved Documents:\n", answer['retrieved_docs'])
    
    
    retrieved_contexts = answer['retrieved_docs']
    
    #this is not an actual output this have been written to test the pipeline
    #response="iphone 16 plus, iphone 16, iphone 15 are best phones under 1,00,000 INR."
    
    context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)
    
    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)
    print("Response Relevancy Score:", relevancy_score)