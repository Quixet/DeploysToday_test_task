from app.rag import get_answer
from app.vector_db_faiss import vector_store
from app.llm import LLM
from dotenv import load_dotenv
import os

vector_store_1 = vector_store()

load_dotenv()
token = HF_TOKEN

llm = LLM(model_name="meta-llama/Llama-3.1-8B-Instruct", token=token)
def chat_response(query: str):
    return get_answer(llm, vector_store_1, query)