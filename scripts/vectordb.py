from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from loader import text_loader
import os
from dotenv import load_dotenv
load_dotenv()

class vectordb:
    def __init__(self):
        self.embedding_model = MistralAIEmbeddings(api_key=os.environ.get('MISTRAL_API_KEY'))


    def create_vector_db(self):
        vectorstore = FAISS.from_documents(text_loader(), self.embedding_model)
        vectorstore.save_local("/Users/navjeetkaur/Downloads/MemoryBot-main/faiss_vector_db")
