from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Central path and runtime configuration for Part III data synthesis."""

    dataset_root: Path = Path("../dataset")
    output_root: Path = Path("./outputs")
    cache_root: Path = Path("./.cache")
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda:0"
    use_fp16: bool = True
    questions_topk: int = 5
    tools_topk: int = 5
    questions_score_threshold: float = 0.5
    tools_score_threshold: float = 0.5
    llm_model: str = "gpt-5.4-mini"
    max_workers: int = 8
    seed: int = 42

    @property
    def part_i_dir(self) -> Path:
        return self.dataset_root / "PartI"

    @property
    def part_ii_dir(self) -> Path:
        return self.dataset_root / "PartII"

    @property
    def part_i_vector_db(self) -> Path:
        return self.dataset_root / "PartI_vector_db"

    @property
    def part_ii_vector_db(self) -> Path:
        return self.dataset_root / "PartII_vector_db"

    @property
    def tool_vector_db(self) -> Path:
        return self.dataset_root / "Tool_vector_db"
