from __future__ import annotations

from langchain_core.documents.base import Document

from .knowledge_base import KnowledgeBase


class Search:
    """Similarity search wrapper used by the synthesis pipeline."""

    def __init__(self, kb: KnowledgeBase, score_threshold: float = 0.5, k: int = 5):
        self.kb = kb
        self.score_threshold = score_threshold
        self.k = k
        self.retriever = self.kb.EmbeddingDB.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": self.score_threshold, "k": self.k},
        )

    def search(self, query: str) -> list[Document]:
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)
        return self.retriever.get_relevant_documents(query)
