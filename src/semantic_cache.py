import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

"""
SEMANTIC CACHE IMPLEMENTATION

Goal:
Traditional caches fail when two users ask the same question using
different wording. This semantic cache solves that by storing queries
based on their vector embeddings rather than raw text.

Key ideas:
1. Queries are converted into embeddings using the same model used
   for document embeddings.
2. Similar queries are detected using cosine similarity.
3. Cache lookup is restricted to relevant semantic clusters
   (built in Task 2) to keep lookup efficient even as the cache grows.
4. The system uses a tunable similarity threshold that determines
   when two queries are considered "close enough".
"""

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Tunable parameter controlling cache strictness
SIMILARITY_THRESHOLD = 0.80

# Number of clusters searched during cache lookup
CLUSTERS_TO_SEARCH = 3

VECTOR_STORE_DIR = "vector_store"

EMBEDDING_FILE = os.path.join(VECTOR_STORE_DIR, "embeddings.npy")
CLUSTER_FILE = os.path.join(VECTOR_STORE_DIR, "cluster_probabilities.npy")
CACHE_FILE = os.path.join(VECTOR_STORE_DIR, "semantic_cache.pkl")


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading document embeddings...")

EMBEDDINGS = np.load(EMBEDDING_FILE)

print("Loading cluster probabilities...")

cluster_probs = np.load(CLUSTER_FILE)

NUM_CLUSTERS = cluster_probs.shape[1]

print(f"Detected {NUM_CLUSTERS} semantic clusters")


# ---------------------------------------------------------
# COMPUTE CLUSTER CENTROIDS (ONLY ONCE)
# ---------------------------------------------------------

print("Computing cluster centroids...")

CLUSTER_CENTROIDS = []

for c in range(NUM_CLUSTERS):

    weights = cluster_probs[:, c][:, None]

    centroid = np.sum(EMBEDDINGS * weights, axis=0) / np.sum(weights)

    CLUSTER_CENTROIDS.append(centroid)

CLUSTER_CENTROIDS = np.array(CLUSTER_CENTROIDS)

print("Cluster centroids ready")


# ---------------------------------------------------------
# CACHE STRUCTURE
# ---------------------------------------------------------

"""
Cache structure:

cluster_cache = {
    cluster_id : [
        {
            "query": original query text
            "embedding": query embedding vector
            "answer": retrieval result
        }
    ]
}

Each cluster maintains its own cache entries.
This dramatically reduces lookup cost when the cache grows large.
"""

if os.path.exists(CACHE_FILE):

    with open(CACHE_FILE, "rb") as f:
        cluster_cache = pickle.load(f)

    print("Existing semantic cache loaded")

else:

    cluster_cache = {i: [] for i in range(NUM_CLUSTERS)}

    print("New semantic cache initialized")


# ---------------------------------------------------------
# CLUSTER PREDICTION
# ---------------------------------------------------------

def predict_clusters(query_embedding, top_k=CLUSTERS_TO_SEARCH):
    """
    Determine which clusters a query belongs to.

    Instead of assigning a query to a single cluster,
    we select the top_k most similar clusters.

    This prevents cache misses when similar queries
    fall near cluster boundaries.
    """

    similarities = cosine_similarity(
        [query_embedding], CLUSTER_CENTROIDS
    )[0]

    top_clusters = np.argsort(similarities)[::-1][:top_k]

    return top_clusters


# ---------------------------------------------------------
# CACHE SEARCH
# ---------------------------------------------------------

def search_cache(query_embedding, cluster_ids):
    """
    Search for similar queries inside the cache.

    Only the relevant clusters are searched, which keeps
    lookup efficient even when the cache grows large.
    """

    best_score = -1
    best_answer = None

    for cid in cluster_ids:

        entries = cluster_cache[cid]

        if len(entries) == 0:
            continue

        embeddings = np.array([e["embedding"] for e in entries])

        similarities = cosine_similarity(
            [query_embedding], embeddings
        )[0]

        idx = np.argmax(similarities)

        if similarities[idx] > best_score:

            best_score = similarities[idx]
            best_answer = entries[idx]["answer"]

    if best_score >= SIMILARITY_THRESHOLD:

        print(f"Cache HIT (similarity={best_score:.3f})")

        return best_answer

    return None


# ---------------------------------------------------------
# ADD NEW CACHE ENTRY
# ---------------------------------------------------------

def add_to_cache(query, embedding, cluster_id, answer):
    """
    Store a query result inside the cache.
    """

    cluster_cache[cluster_id].append(
        {
            "query": query,
            "embedding": embedding,
            "answer": answer,
        }
    )

    print("Query stored in cache")


# ---------------------------------------------------------
# SAVE CACHE
# ---------------------------------------------------------

def save_cache():

    with open(CACHE_FILE, "wb") as f:

        pickle.dump(cluster_cache, f)


# ---------------------------------------------------------
# SIMULATED SEMANTIC SEARCH
# ---------------------------------------------------------

def retrieve_documents(query_embedding, top_k=3):
    """
    Simulated semantic search over the corpus.

    In a real production system this would query a
    vector database like FAISS or Pinecone.
    """

    similarities = cosine_similarity(
        [query_embedding], EMBEDDINGS
    )[0]

    indices = np.argsort(similarities)[::-1][:top_k]

    return indices.tolist()


# ---------------------------------------------------------
# MAIN QUERY PIPELINE
# ---------------------------------------------------------

def process_query(query):

    print("\nUser Query:", query)

    # Step 1: convert query to embedding
    query_embedding = model.encode(query)

    # Step 2: determine relevant clusters
    cluster_ids = predict_clusters(query_embedding)

    print("Candidate clusters:", cluster_ids)

    # Step 3: search semantic cache
    cached_answer = search_cache(query_embedding, cluster_ids)

    if cached_answer is not None:

        return cached_answer

    # Step 4: compute answer using semantic search
    print("Cache MISS → running semantic search")

    result = retrieve_documents(query_embedding)

    # Step 5: store result in cache
    add_to_cache(query, query_embedding, cluster_ids[0], result)

    save_cache()

    return result


# ---------------------------------------------------------
# DEMONSTRATION
# ---------------------------------------------------------

if __name__ == "__main__":

    queries = [

        "How does rocket propulsion work?",
        "Explain rocket engines",

        "What causes earthquakes?",
        "Why do earthquakes happen?"
    ]

    for q in queries:

        result = process_query(q)

        print("Search result document IDs:", result)