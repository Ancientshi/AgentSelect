from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from .config import ProjectConfig
from .embeddings import BGEEmbedding
from .io_utils import load_json


@dataclass
class LoadedDocument:
    text: str
    metadata: dict[str, Any]


def load_questions(question_json: str | Path) -> list[LoadedDocument]:
    questions = load_json(question_json)
    docs: list[LoadedDocument] = []
    for idx, (key, value) in enumerate(questions.items()):
        text = value.get("input", "") if isinstance(value, dict) else str(value)
        docs.append(LoadedDocument(text=text, metadata={"str_index": idx, "key": key}))
    return docs


def load_tools(tool_json: str | Path) -> list[LoadedDocument]:
    tools = load_json(tool_json)
    docs: list[LoadedDocument] = []

    for idx, (tool_name, value) in enumerate(tools.items()):
        if isinstance(value, dict):
            description = value.get("description", "")
        else:
            description = value

        # keep the same behavior as the original f-string implementation
        description = str(description)

        docs.append(
            LoadedDocument(
                text=description,
                metadata={
                    "str_index": idx,
                    "tool_name": str(tool_name),
                },
            )
        )

    return docs


class VectorDatabase:
    """Thin Chroma wrapper, kept close to the original implementation."""

    def __init__(self, embeddings, persist_directory: str | Path):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.vectordb = self._create_chroma()

    def _create_chroma(self):
        return Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": self.embeddings.get_sim_cal()},
        )

    def reset(self) -> None:
        try:
            self.vectordb.delete_collection()
        except Exception:
            pass
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.vectordb = self._create_chroma()

    def add_documents(
        self,
        documents: Iterable[LoadedDocument | dict[str, Any]],
        recreate: bool = False,
        batch_size: int = 512,
    ) -> None:
        if recreate:
            self.reset()

        docs: list[Document] = []
        for item in documents:
            if isinstance(item, dict):
                text = item["text"]
                metadata = item.get("metadata", {})
            else:
                text = item.text
                metadata = item.metadata
            docs.append(Document(page_content=text, metadata=metadata))

        if not docs:
            return

        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            self.vectordb.add_documents(docs[start:end])
            print(f"Added documents {start + 1}-{end} / {len(docs)}")

        if hasattr(self.vectordb, "persist"):
            self.vectordb.persist()

    def count(self) -> int:
        return self.vectordb._collection.count()


class KnowledgeBase:
    def __init__(self, vectordb: VectorDatabase):
        self.EmbeddingDB = vectordb

    def process_questions(self, json_path: str | Path, recreate: bool = False) -> None:
        self.EmbeddingDB.add_documents(load_questions(json_path), recreate=recreate)

    def process_tools(self, json_path: str | Path, recreate: bool = False) -> None:
        self.EmbeddingDB.add_documents(load_tools(json_path), recreate=recreate)


def build_bge_embedding(config: ProjectConfig) -> BGEEmbedding:
    model = BGEM3FlagModel(
        config.model_name,
        use_fp16=config.use_fp16,
        device=config.device,
    )
    return BGEEmbedding(model)


def build_index(
    kind: Literal["part_i_questions", "part_ii_questions", "tools"],
    config: ProjectConfig,
    recreate: bool = False,
) -> VectorDatabase:
    embeddings = build_bge_embedding(config)
    if kind == "part_i_questions":
        db = VectorDatabase(embeddings, config.part_i_vector_db)
        KnowledgeBase(db).process_questions(config.part_i_dir / "questions" / "merge.json", recreate=recreate)
    elif kind == "part_ii_questions":
        db = VectorDatabase(embeddings, config.part_ii_vector_db)
        KnowledgeBase(db).process_questions(config.part_ii_dir / "questions" / "merge.json", recreate=recreate)
    elif kind == "tools":
        db = VectorDatabase(embeddings, config.tool_vector_db)
        tool_file = config.dataset_root / "Tools" / "merge.json"

        if not tool_file.exists():
            raise FileNotFoundError(f"Tool file not found: {tool_file}")

        print(f"[INFO] Using tool file: {tool_file}")
        KnowledgeBase(db).process_tools(tool_file, recreate=recreate)
    else:
        raise ValueError(f"Unknown index kind: {kind}")
    
    return db