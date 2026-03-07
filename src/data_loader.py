import os
from typing import List, Tuple
from text_preprocessing import clean_text


DATA_PATH = "data/20_newsgroups"


def load_newsgroup_data() -> Tuple[List[str], List[str]]:
    """
    Loads the 20 Newsgroups dataset from disk.

    The dataset is organised as:

        category_name/
            document1
            document2
            ...

    Each folder name represents a topic label.

    Even though we later perform unsupervised clustering,
    we keep the labels as metadata. This allows us to:

    - evaluate cluster coherence
    - perform filtered retrieval
    - inspect semantic boundaries between clusters
    """

    documents = []
    labels = []

    for category in os.listdir(DATA_PATH):

        category_path = os.path.join(DATA_PATH, category)

        if not os.path.isdir(category_path):
            continue

        for file_name in os.listdir(category_path):

            file_path = os.path.join(category_path, file_name)

            try:
                with open(file_path, "r", encoding="latin1") as f:
                    raw_text = f.read()

                    cleaned = clean_text(raw_text)

                    # Discard extremely short documents
                    # They usually contain signatures or empty replies
                    if len(cleaned) < 50:
                        continue

                    documents.append(cleaned)
                    labels.append(category)

            except Exception:
                continue

    return documents, labels


if __name__ == "__main__":

    docs, labels = load_newsgroup_data()

    print("Total documents:", len(docs))
    print("Example document:\n")
    print(docs[0][:500])