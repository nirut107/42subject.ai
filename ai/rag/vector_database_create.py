import sqlite3
import faiss
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer
import networkx as nx
import json
import re
import requests

VPS_URL = "http://103.195.7.156:8000/chat"


def call_ollama_via_vps(prompt: str) -> str:
    res = requests.post(
        VPS_URL,
        json={
            "prompt": prompt
        },
        timeout=120,
    )
    res.raise_for_status()
    return res.json()

response = call_ollama_via_vps("hi")
print(response)

def get_db():
    return sqlite3.connect("../database/dev.db")

conn = get_db()
cursor = conn.cursor()

cursor.execute("""
    SELECT id, name, content,summary , core, rank
    FROM file_data
""")

rows = cursor.fetchall()
conn.close()

print(f"Loaded {len(rows)} documents from database")


def chunk_text(text, chunk_size=500, overlap=80):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para[-overlap:] + " " + para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks





documents = []

for row in rows:
    file_id, name, content,summary , core, rank = row

    chunks = chunk_text(content)

    if summary :
        for i, chunk in enumerate(chunks):
            documents.append({
                "chunk_id": f"{file_id}_{i}",
                "file_id": file_id,
                "file_name": name,
                "core": core,
                "rank": rank,
                "text": name + " " + chunk,
            })
    
        documents.append({
            "chunk_id": f"{file_id}_{-1}",
                "file_id": file_id,
                "file_name": name,
                "core": core,
                "rank": rank,
                "text": name + " " + summary ,
            })
    

print(f"Created {len(documents)} chunks")

model = SentenceTransformer("all-MiniLM-L6-v2")

# texts = [doc["text"] for doc in documents]
# embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# # Normalize for cosine similarity
# faiss.normalize_L2(embeddings)

# dim = embeddings.shape[1]
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings)

# print("FAISS index size:", index.ntotal)



# faiss.write_index(index, "faiss_index.bin")


# with open("faiss_metadata.json", "w", encoding="utf-8") as f:
#     json.dump(documents, f, ensure_ascii=False, indent=2)

EXTRACTION_PROMPT = """
You are extracting a knowledge graph from a software project specification.

IMPORTANT:
- The text belongs to the project named: "{project_name}"
- Every extracted entity MUST belong to this project
- DO NOT extract generic concepts unless they are explicitly used in this project
- DO NOT extract entities from other projects

Entity types:
- Project
- Concept
- Parameter
- Requirement
- Action
- Component

Relation types:
- BELONGS_TO
- REQUIRES
- FORBIDS
- USES
- HAS_PARAMETER
- CONSISTS_OF
- STATE_TRANSITION
- LOG_FORMAT

Rules:
- Create ONE Project entity with id: "project_{project_name}"
- Every other entity MUST have a BELONGS_TO relation pointing to the Project
- Entity ids must be lowercase snake_case
- Return VALID JSON ONLY (no markdown, no explanation)

JSON format:
{{
  "entities": [
    {{ "id": "...", "type": "...", "name": "..." }}
  ],
  "relations": [
    {{ "source": "...", "relation": "...", "target": "...", "evidence": "..." }}
  ]
}}

Text:
\"\"\"
{chunk_text}
\"\"\"
"""






def extract_kg_from_chunk(chunk_text,project_name):
    # response = client.chat.completions.create(
    #     model="gpt-4.1-mini",   # fast + cheap
    #     messages=[
    #         {"role": "system", "content": EXTRACTION_PROMPT},
    #         {"role": "user", "content": chunk_text}
    #     ],
    #     temperature=0
    # )
    prompt = EXTRACTION_PROMPT.format(
        project_name=project_name,
        chunk_text=chunk_text
    )
    response = call_ollama_via_vps(prompt)
    answer = response["answer"]

    # content = response.choices[0].message.content
    return answer

def clean_llm_json(text):
    """
    Removes markdown code fences and extracts JSON safely.
    """
    if not text:
        return None

    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # Trim whitespace
    text = text.strip()

    # Ensure it starts with { and ends with }
    if not text.startswith("{") or not text.endswith("}"):
        print("⚠️ LLM output is not pure JSON")
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("⚠️ JSON parsing error:", e)
        return None

def parse_kg_json(json_text):
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON from LLM, skipping chunk")
        return None



G = nx.DiGraph()

def add_to_graph(graph, kg_data, source_chunk):
    if not kg_data:
        return

    # Add entities
    for ent in kg_data.get("entities", []):
        node_id = ent["id"]

        if not graph.has_node(node_id):
            graph.add_node(
                node_id,
                type=ent["type"],
                name=ent["name"],
                sources=[source_chunk]
            )
        else:
            graph.nodes[node_id]["sources"].append(source_chunk)

    # Add relations
    for rel in kg_data.get("relations", []):
        graph.add_edge(
            rel["source"],
            rel["target"],
            relation=rel["relation"],
            evidence=rel["evidence"],
            source_chunk=source_chunk
        )

for chunk in documents:
    # print(f"Processing chunk {chunk["chunk_id"]}")

    raw_json = extract_kg_from_chunk(chunk["text"], chunk["file_name"])
    # raw_json = extract_kg_cached(chunk["chunk_id"], chunk["text"])
    # raw_json = safe_llm_call(extract_kg_cached, chunk_text)
    # raw_json = safe_llm_call(extract_kg_from_chunk, chunk["text"], chunk["file_name"])


    # print(raw_json)
    # kg_data = parse_kg_json(raw_json)
    kg_data = clean_llm_json(raw_json)

    add_to_graph(G, kg_data, chunk["chunk_id"])


import pickle

with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_node_embeddings(graph):
    node_ids = []
    texts = []

    for node, data in graph.nodes(data=True):
        text = f"{data.get('name','')} ({data.get('type','')})"
        node_ids.append(node)
        texts.append(text)

    embeddings = model.encode(texts, convert_to_numpy=True)
    return node_ids, embeddings

node_ids, node_embeddings = build_node_embeddings(G)

np.save("node_embeddings.npy", node_embeddings)

with open("node_ids.json", "w") as f:
    json.dump(node_ids, f)