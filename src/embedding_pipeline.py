import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_loader import load_newsgroup_data


def generate_embeddings():
    """
    Convert documents into dense vector embeddings.

    Embedding Model Choice:
    -----------------------
    We use 'all-MiniLM-L6-v2' from SentenceTransformers.

    Reasons:

    1. Strong semantic performance for retrieval tasks
    2. Lightweight (~80MB) suitable for local deployment
    3. Produces 384 dimensional embeddings
    4. Fast inference even without GPU

    This makes it a good fit for a lightweight semantic
    search system like the one required in this assignment.
    """

    documents, labels = load_newsgroup_data()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=64
    )

    embeddings = np.array(embeddings)

    print("Embedding shape:", embeddings.shape)

    return embeddings, documents, labels