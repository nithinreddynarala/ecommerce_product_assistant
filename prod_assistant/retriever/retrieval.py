import os
from langchain_astradb import AstraDBVectorStore
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
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
         
        required_vars=["GROQ_API_KEY","HF_TOKEN","ASTRA_DB_API_ENDPOINT","ASTRA_DB_APPLICATION_TOKEN","ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    def load_retriever(self):
        """_summary_
        """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            
            self.vstore = AstraDBVectorStore(
            embedding= self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.astra_db_api_endpoint,
            token=self.astra_db_application_token,
            namespace=self.astra_db_keyspace,
        )

        if not self.retriever_instance:
            top_k=self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            self.retriever_instance=self.vstore.as_retriever(search_kwargs={"k":top_k})
        return self.retriever_instance
            
            
            
            
            
    def call_retriever(self,query):
        """_summary_
        """
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output
    
if __name__=='__main__':
    user_query = "Can you suggest good budget iPhone under 1,00,00 INR?"
    
    retriever_obj = Retriever()
    

    retrieved_docs = retriever_obj.call_retriever(user_query)
    
    

    
    
    
    for idx, doc in enumerate(retrieved_docs):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")