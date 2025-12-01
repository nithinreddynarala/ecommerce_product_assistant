import os
from langchain_astradb import AstraDBVectorStore
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from evaluation.ragas import evaluate_context_precision, evaluate_response_relevancy
# Add the project root to the Python path for direct script execution
# project_root = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(project_root))

class Retriever:
    def __init__(self):
        """_summary_
        """
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever_instance = None
    
    def _load_env_variables(self):
        """_summary_
        """
        load_dotenv()
         
        required_vars = ["GROQ_API_KEY","HF_TOKEN", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    def load_retriever(self):
        """_summary_
        """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            
            self.vstore =AstraDBVectorStore(
                embedding= self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
                )
        if not self.retriever_instance:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3

            #self.retriever_instance=self.vstore.as_retriever(search_kwargs={"k":top_k})
            
            self.retriever_instance=self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k,
                                "fetch_k": 20,
                                "lambda_mult": 0.7,
                                "score_threshold": 0.6
                               })
            print("Retriever loaded successfully.")
            
            '''llm = self.model_loader.load_llm()
            
            compressor=LLMChainFilter.from_llm(llm)
            
            self.retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=self.retriever_instance
            )'''
            
        return self.retriever_instance
            
    def call_retriever(self,query):
        """_summary_
        """
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output
    
if __name__=='__main__':
    user_query = "what is the price of samsung s24?"
    
    retriever_obj = Retriever()
    
    retrieved_docs = retriever_obj.call_retriever(user_query)

    print(retrieved_docs)

    
    
    def _format_docs(docs):
        if not docs:
            return "No relevant documents found."

        formatted_chunks = []

        
        meta = docs.metadata or {}
        formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{docs.page_content.strip()}"
            )
        formatted_chunks.append(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    
    retrieved_contexts = [_format_docs(doc) for doc in retrieved_docs]
    
    #this is not an actual output this have been written to test the pipeline
    response="The price of samsung s24 is ₹45,999."
    
    context_score = evaluate_context_precision(user_query,response,retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query,response,retrieved_contexts)
    
    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)
    print("Response Relevancy Score:", relevancy_score)
    

    
    
    
    # for idx, doc in enumerate(results, 1):
    #     print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")