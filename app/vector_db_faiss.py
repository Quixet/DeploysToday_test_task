import faiss
import numpy as np

index = faiss.read_index("C:/Users/Acer/Desktop/DeploysToday_test_task/app/cocktail_index.faiss")

def vector_store():
    return index
