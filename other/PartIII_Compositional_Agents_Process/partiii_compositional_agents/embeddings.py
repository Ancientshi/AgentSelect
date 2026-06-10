from __future__ import annotations

import asyncio
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings


class MyOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embedding wrapper with Chroma similarity metadata."""

    def get_sim_cal(self) -> str:
        return "cosine"


class MiniLMEmbedding(Embeddings):
    """SentenceTransformer wrapper used by LangChain vector stores."""

    def __init__(self, base_model):
        self.model = base_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    def get_sim_cal(self) -> str:
        return "cosine"


class BGEEmbedding(Embeddings):
    """BGE-M3 dense embedding wrapper.

    The returned vectors use BGE-M3 dense vectors and are stored in Chroma with
    inner-product similarity, matching the original Part III synthesis scripts.
    """

    def __init__(self, base_model, max_length: int = 1024, batch_size: int = 4):
        self.model = base_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = getattr(base_model, "tokenizer", None)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.encode(
            [text],
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]
        return embeddings[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    def get_sim_cal(self) -> str:
        return "ip"
