from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_relevant_nodes(question, node_ids, node_embeddings, top_k=3):
    q_emb = model.encode([question], convert_to_numpy=True)

    sims = cosine_similarity(q_emb, node_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]

    return [(node_ids[i], sims[i]) for i in top_indices]


def expand_graph(graph, seed_nodes, hops=1):
    expanded = set(seed_nodes)

    for _ in range(hops):
        new_nodes = set()
        for node in expanded:
            new_nodes.update(graph.successors(node))
            new_nodes.update(graph.predecessors(node))
        expanded |= new_nodes

    return expanded


def build_graph_context(graph, nodes):
    context_lines = []

    for node in nodes:
        node_data = graph.nodes[node]
        context_lines.append(
            f"[ENTITY] {node_data['name']} ({node_data['type']})"
        )

        for _, target, edge_data in graph.out_edges(node, data=True):
            context_lines.append(
                f"  - {edge_data['relation']} → {graph.nodes[target]['name']}"
                f" | evidence: {edge_data.get('evidence','')}"
            )

    return "\n".join(context_lines)

def graph_rag_retrieve(question, graph, node_ids, node_embeddings):
    # 1. Semantic node retrieval
    seed_nodes = retrieve_relevant_nodes(
        question, node_ids, node_embeddings, top_k=3
    )

    seed_node_ids = [n for n, _ in seed_nodes]

    # 2. Graph expansion
    expanded_nodes = expand_graph(
        graph, seed_node_ids, hops=1
    )

    # 3. Build context
    context = build_graph_context(graph, expanded_nodes)

    return {
        "seed_nodes": seed_nodes,
        "expanded_nodes": list(expanded_nodes),
        "context": context
    }

# question = "What parameters control philosopher behavior?"
# question = "what is trancendence"


# result = graph_rag_retrieve(
#     question, G, node_ids, node_embeddings
# )

# print("=== GRAPH CONTEXT ===")
# print(result["context"])
