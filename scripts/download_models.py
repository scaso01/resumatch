"""Pre-download ML models for Docker build caching."""

from sentence_transformers import SentenceTransformer

print("Downloading all-MiniLM-L6-v2...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Done.")
