import re


def clean_text(text: str) -> str:
    """
    Clean raw newsgroup documents.

    The 20 Newsgroups dataset contains several noisy components:
    - Email headers (From:, Subject:, Organization:)
    - Quoted replies starting with ">"
    - Long signatures at the bottom
    - Excessive whitespace and special characters

    For semantic search, these components introduce noise and
    reduce embedding quality. Therefore we remove them.

    We deliberately KEEP the main message body because
    it carries the semantic signal required for clustering
    and retrieval.
    """

    # Remove email headers (everything before first blank line)
    text = re.split(r"\n\s*\n", text, maxsplit=1)[-1]

    # Remove quoted replies
    text = re.sub(r">.*", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()