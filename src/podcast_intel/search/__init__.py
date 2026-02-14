"""
Semantic search module using embeddings and vector store.

Provides multilingual semantic search across podcast transcripts
using BGE-M3 embeddings and ChromaDB vector store.
"""

from podcast_intel.search.embedder import Embedder
from podcast_intel.search.vector_store import VectorStore
from podcast_intel.search.query import QueryEngine

__all__ = ["Embedder", "VectorStore", "QueryEngine"]
