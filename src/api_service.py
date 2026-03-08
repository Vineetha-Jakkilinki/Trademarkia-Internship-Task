"""
FastAPI service for Semantic Search + Semantic Cache.

This exposes three endpoints:
POST /query
GET /cache/stats
DELETE /cache
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------

app = FastAPI(title="Semantic Search API")


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = None

@app.on_event("startup")
def load_model():
    global model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------
# Load Vector Store Data
# ---------------------------------------------------

print("Loading embeddings...")
doc_embeddings = np.load("vector_store/embeddings.npy")

print("Loading cluster probabilities...")
cluster_probs = np.load("vector_store/cluster_probabilities.npy")

print("Loading documents...")
with open("vector_store/documents.json") as f:
    documents = json.load(f)


# ---------------------------------------------------
# Compute cluster centroids
# ---------------------------------------------------

NUM_CLUSTERS = cluster_probs.shape[1]
cluster_centroids = []

print("Computing cluster centroids...")

for i in range(NUM_CLUSTERS):

    cluster_docs = doc_embeddings[cluster_probs.argmax(axis=1) == i]

    if len(cluster_docs) > 0:
        centroid = np.mean(cluster_docs, axis=0)
    else:
        centroid = np.zeros(doc_embeddings.shape[1])

    cluster_centroids.append(centroid)

cluster_centroids = np.array(cluster_centroids)

print("Cluster centroids ready")


# ---------------------------------------------------
# Load Semantic Cache
# ---------------------------------------------------

try:
    with open("vector_store/semantic_cache.pkl", "rb") as f:
        semantic_cache = pickle.load(f)

    print("Existing cache loaded")

except:
    semantic_cache = {}
    print("Starting with empty cache")


# ---------------------------------------------------
# Cache Statistics
# ---------------------------------------------------

hit_count = 0
miss_count = 0

SIMILARITY_THRESHOLD = 0.80


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------------------
# Helper: Semantic Search
# ---------------------------------------------------

def run_semantic_search(query_embedding, dominant_cluster, top_k=3):

    cluster_indices = np.where(cluster_probs.argmax(axis=1) == dominant_cluster)[0]

    cluster_embeddings = doc_embeddings[cluster_indices]

    similarities = cosine_similarity([query_embedding], cluster_embeddings)[0]

    top_local = np.argsort(similarities)[-top_k:][::-1]

    top_indices = cluster_indices[top_local]

    return top_indices


# ---------------------------------------------------
# POST /query
# ---------------------------------------------------

@app.post("/query")
def query_endpoint(request: QueryRequest):

    global hit_count, miss_count

    query = request.query

    print("\nUser Query:", query)

    query_embedding = model.encode(query)

    # -----------------------------
    # Determine dominant cluster
    # -----------------------------

    cluster_scores = cosine_similarity([query_embedding], cluster_centroids)[0]

    dominant_cluster = int(np.argmax(cluster_scores))

    # -----------------------------
    # Cache lookup
    # -----------------------------

    best_match = None
    best_similarity = 0

    for cached_query, data in semantic_cache.items():

        # Skip corrupted cache entries
        if not isinstance(data, dict) or "embedding" not in data:
            continue

        similarity = cosine_similarity(
            [query_embedding],
            [data["embedding"]]
        )[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cached_query


    # -----------------------------
    # Cache HIT
    # -----------------------------

    if best_similarity >= SIMILARITY_THRESHOLD:

        hit_count += 1

        print("Cache HIT")

        cached_data = semantic_cache[best_match]

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": best_match,
            "similarity_score": float(best_similarity),
            "result": cached_data["result"],
            "dominant_cluster": dominant_cluster
        }


    # -----------------------------
    # Cache MISS
    # -----------------------------

    miss_count += 1

    print("Cache MISS → running search")

    results = run_semantic_search(query_embedding, dominant_cluster)

    result_texts = [documents[i] for i in results]


    # Store in cache
    semantic_cache[query] = {
        "embedding": query_embedding,
        "result": result_texts
    }

    with open("vector_store/semantic_cache.pkl", "wb") as f:
        pickle.dump(semantic_cache, f)


    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(best_similarity),
        "result": result_texts,
        "dominant_cluster": dominant_cluster
    }


# ---------------------------------------------------
# GET /cache/stats
# ---------------------------------------------------

@app.get("/cache/stats")
def cache_stats():

    total_entries = len(semantic_cache)

    total_queries = hit_count + miss_count

    hit_rate = 0

    if total_queries > 0:
        hit_rate = hit_count / total_queries

    return {
        "total_entries": total_entries,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": round(hit_rate, 3)
    }


# ---------------------------------------------------
# DELETE /cache
# ---------------------------------------------------

@app.delete("/cache")
def clear_cache():

    global semantic_cache, hit_count, miss_count

    semantic_cache = {}

    hit_count = 0
    miss_count = 0

    with open("vector_store/semantic_cache.pkl", "wb") as f:
        pickle.dump({}, f)

    return {"message": "Cache cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "src.api_service:app",
        host="0.0.0.0",
        port=port
    )