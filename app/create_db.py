import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


cocktails_df = pd.read_csv('C:/Users/Acer/Desktop/DeploysToday_test_task/app/final_cocktails.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

cocktails_df['ingredients_cleaned'] = cocktails_df['ingredients'].apply(
    lambda x: ', '.join(eval(x)) if isinstance(x, str) else ''
)
embeddings = model.encode(cocktails_df['ingredients_cleaned'].tolist())

embeddings = np.array(embeddings, dtype='float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, 'cocktail_index.faiss')