import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Retrieval part ---
with open("documents/data.txt", "r") as file:
    documents = [line.strip() for line in file.readlines() if line.strip()]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype("float32"))

question = input("Ask Question: ")
query_embedding = model.encode([question])

k = min(3, len(documents))
distances, indices = index.search(
    np.array(query_embedding).astype("float32"), k=k
)

retrieved_docs = [documents[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

print("\nRetrieved Context:")
print(context)

# --- Generation part (OpenRouter) ---
api_key = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "RAG Pipeline App",
    },
)

prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": prompt}],
)

print("\nAnswer:")
print(response.choices[0].message.content)