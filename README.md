# Semantic Search API

This project implements a **Semantic Search system with Semantic Cache** using Sentence Transformers and FastAPI.
It retrieves the most relevant documents for a query using vector embeddings and optimizes repeated queries using caching.

## Live API

Live API Endpoint:
https://vineetha-jakkilinki-semantic-search-api.hf.space/docs

## Features

* Semantic Search using Sentence Transformers
* Cluster-based search optimization
* Semantic caching for repeated queries
* FastAPI backend
* Deployed API

## Endpoints

POST `/query` – Perform semantic search for a query
GET `/cache/stats` – View cache statistics
DELETE `/cache` – Clear the semantic cache

## Tech Stack

* Python
* FastAPI
* Sentence Transformers
* NumPy
* Scikit-learn

## Author

Vineetha Jakkilinki
