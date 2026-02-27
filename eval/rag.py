"""
RAG module using LlamaIndex for document retrieval.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class RAGRetriever:
    """RAG retriever using LlamaIndex."""

    def __init__(
        self,
        corpus_paths: list[str],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        cache_dir: str = ".rag_cache",
    ):
        self.corpus_paths = [Path(p) for p in corpus_paths]
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.index = None
        self.chunks: list[str] = []

    def _get_cache_key(self) -> str:
        corpus_str = ",".join(str(p) for p in sorted(self.corpus_paths))
        return hashlib.md5(f"{corpus_str}:{self.chunk_size}".encode()).hexdigest()

    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load the vector index."""
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Step 1: Try loading from cache
        if not force_rebuild and cache_file.exists():
            print("Loading RAG index from cache...")
            with open(cache_file) as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            print(f"Loaded {len(self.chunks)} chunks")
            return

        print("Building RAG index...")

        # Step 2: Configure LlamaIndex settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Step 3: Load documents from corpus paths
        documents = []
        for path in self.corpus_paths:
            if path.exists():
                if path.is_file():
                    docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
                else:
                    docs = SimpleDirectoryReader(str(path)).load_data()
                documents.extend(docs)

        print(f"Loaded {len(documents)} documents")

        # Step 4: Create vector index from documents
        self.index = VectorStoreIndex.from_documents(documents)
        self.chunks = [node.text for node in self.index.docstore.docs.values()]

        # Step 5: Save index to cache for future use
        with open(cache_file, "w") as f:
            json.dump({"chunks": self.chunks}, f)

        print(f"RAG index built: {len(self.chunks)} chunks")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve top-k chunks for a query."""
        if self.index is None:
            # Force rebuild since cache only stores chunks, not the index
            self.build_index(force_rebuild=True)
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        return [node.text for node in nodes]

    def get_context(
        self,
        query: str,
        k: int = 3,
        template: str = "- {text}",
        separator: str = "\n",
    ) -> str:
        """Get formatted context for a query."""
        chunks = self.retrieve(query, k)
        return separator.join(template.format(text=c) for c in chunks)


class RAGManager:
    """Manager for RAG retrievers."""

    def __init__(self):
        self.retrievers: dict[str, RAGRetriever] = {}

    def get_retriever(
        self,
        name: str,
        corpus_paths: list[str],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> RAGRetriever:
        if name not in self.retrievers:
            self.retrievers[name] = RAGRetriever(
                corpus_paths=corpus_paths,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.retrievers[name].build_index()
        return self.retrievers[name]


def augment_sample_with_rag(
    sample: dict[str, Any],
    rag_config: dict[str, Any],
    retriever: RAGRetriever,
) -> dict[str, Any]:
    """Augment a sample with RAG context."""
    query_field = rag_config.get("query_field", "text")
    context_field = rag_config.get("context_field", "context")
    query = sample.get(query_field, "")

    if not query:
        return sample

    context = retriever.get_context(
        query=query,
        k=rag_config.get("context_k", 3),
        template=rag_config.get("context_template", "- {text}"),
        separator=rag_config.get("context_separator", "\n"),
    )

    augmented = dict(sample)
    augmented[context_field] = context
    return augmented
