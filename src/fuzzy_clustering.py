import numpy as np
import json
import os

from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer


VECTOR_DIR = "vector_store"


def load_vector_store():
    """
    Load embeddings and associated metadata created in Task-1.
    """

    embeddings = np.load(f"{VECTOR_DIR}/embeddings.npy")

    with open(f"{VECTOR_DIR}/documents.json", "r") as f:
        documents = json.load(f)

    with open(f"{VECTOR_DIR}/metadata.json", "r") as f:
        metadata = json.load(f)

    return embeddings, documents, metadata


# ------------------------------------------------------------
# Step 1: Determine optimal number of clusters
# ------------------------------------------------------------

def find_optimal_clusters(embeddings):
    """
    We determine the number of clusters using BIC
    (Bayesian Information Criterion).

    BIC balances:
        model fit vs model complexity

    Lower BIC values indicate a better clustering model
    without unnecessary complexity.

    This provides evidence-based cluster selection
    rather than arbitrarily choosing a number.
    """

    cluster_range = range(5, 41, 5)
    bic_scores = []

    print("\nEvaluating cluster numbers using BIC...\n")

    for k in cluster_range:

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42
        )

        gmm.fit(embeddings)

        bic = gmm.bic(embeddings)

        bic_scores.append(bic)

        print(f"Clusters: {k} | BIC Score: {bic}")

    best_k = cluster_range[np.argmin(bic_scores)]

    print("\nOptimal cluster number based on BIC:", best_k)

    return best_k


# ------------------------------------------------------------
# Step 2: Train fuzzy clustering model
# ------------------------------------------------------------

def perform_fuzzy_clustering(embeddings, n_clusters):
    """
    Train Gaussian Mixture Model for fuzzy clustering.

    Unlike KMeans, GMM assigns probabilities instead
    of hard labels.

    Output example:
        Doc1 → [0.2, 0.5, 0.3]

    meaning the document belongs to:
        20% cluster1
        50% cluster2
        30% cluster3
    """

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        random_state=42
    )

    gmm.fit(embeddings)

    cluster_probs = gmm.predict_proba(embeddings)

    print("\nCluster probability matrix shape:", cluster_probs.shape)

    return gmm, cluster_probs


# ------------------------------------------------------------
# Step 3: Analyze cluster meaning
# ------------------------------------------------------------

def analyze_clusters(cluster_probs, documents, n_clusters):
    """
    To interpret cluster semantics we inspect
    the documents with highest probability
    within each cluster.

    This helps verify whether clusters correspond
    to meaningful topics.
    """

    print("\n\nCluster interpretation:\n")

    for c in range(n_clusters):

        doc_indices = np.argsort(cluster_probs[:, c])[-5:]

        print(f"\nCluster {c} top documents:\n")

        for idx in doc_indices:

            snippet = documents[idx][:200]

            print("-", snippet.replace("\n", " "), "\n")


# ------------------------------------------------------------
# Step 4: Identify boundary documents
# ------------------------------------------------------------

def find_boundary_documents(cluster_probs, documents):
    """
    Boundary documents are those where the model
    is uncertain between clusters.

    These documents have similar probability
    across multiple clusters.

    Example:
        [0.4, 0.35, 0.25]

    These are particularly interesting because they
    lie between semantic themes.
    """

    entropy_scores = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10), axis=1)

    boundary_indices = np.argsort(entropy_scores)[-10:]

    print("\n\nBoundary documents (high cluster uncertainty):\n")

    for idx in boundary_indices:

        snippet = documents[idx][:200]

        print("-", snippet.replace("\n", " "), "\n")


# ------------------------------------------------------------
# Step 5: Save clustering output
# ------------------------------------------------------------

def save_cluster_results(cluster_probs):
    """
    Save fuzzy cluster memberships.

    Each document receives a probability distribution
    across clusters.

    This data will later be used in:
        - semantic cache optimisation
        - cluster-based retrieval
    """

    np.save(f"{VECTOR_DIR}/cluster_probabilities.npy", cluster_probs)

    print("\nCluster probabilities saved.")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def run_clustering_pipeline():

    embeddings, documents, metadata = load_vector_store()

    n_clusters = find_optimal_clusters(embeddings)

    model, cluster_probs = perform_fuzzy_clustering(embeddings, n_clusters)

    analyze_clusters(cluster_probs, documents, n_clusters)

    find_boundary_documents(cluster_probs, documents)

    save_cluster_results(cluster_probs)


if __name__ == "__main__":
    run_clustering_pipeline()