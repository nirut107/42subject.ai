import faiss
import json
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")
documents = json.load(open("faiss_metadata.json"))

def normal_rag_search(question, k=20):
    q_emb = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        doc = documents[idx]
        results.append({
            "text": doc["text"],
            "file_name": doc["file_name"],
            "core": doc["core"],
            "score": float(score)
        })

    return results

# question = "What happens if a student does not meet attendance requirements?"

# results = normal_rag_search(question)

# for r in results:
#     print(f"[{r['score']:.2f}] ({r['core']}) {r['file_name']}")
#     print(r["text"])
#     print("----")
