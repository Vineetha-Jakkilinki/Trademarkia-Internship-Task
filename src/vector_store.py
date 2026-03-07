import numpy as np
import json
import os

from embedding_pipeline import generate_embeddings


VECTOR_DIR = "vector_store"


def build_vector_store():
    """
    Persist embeddings and metadata to disk.

    We store three components:

    1. embeddings.npy
        Dense vector representations

    2. documents.json
        Original cleaned text

    3. metadata.json
        Contains labels and document ids

    This separation allows efficient vector search
    while still supporting filtered retrieval
    based on metadata such as category labels.
    """

    os.makedirs(VECTOR_DIR, exist_ok=True)

    embeddings, documents, labels = generate_embeddings()

    # Save embeddings
    np.save(f"{VECTOR_DIR}/embeddings.npy", embeddings)

    # Save documents
    with open(f"{VECTOR_DIR}/documents.json", "w") as f:
        json.dump(documents, f)

    # Save metadata
    metadata = [
        {"doc_id": i, "label": labels[i]}
        for i in range(len(labels))
    ]

    with open(f"{VECTOR_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Vector store created successfully.")


if __name__ == "__main__":
    build_vector_store()