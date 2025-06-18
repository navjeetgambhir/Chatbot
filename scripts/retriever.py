from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
import os



def create_retriever():
        os.environ.get('HF_TOKEN')
        api_key = os.environ.get('MISTRAL_API_KEY') 
        embedding_model = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
        vectorstore = FAISS.load_local("/Users/navjeetkaur/Downloads/MemoryBot-main/faiss_vector_db", 
                                   embedding_model,allow_dangerous_deserialization=True)  
        retriever=vectorstore.as_retriever(search_type="similarity_score_threshold",
                                            search_kwargs={'score_threshold': 0.5 }) 
        return retriever




   




