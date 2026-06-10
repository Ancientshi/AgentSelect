#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from partiii_compositional_agents.config import ProjectConfig
from partiii_compositional_agents.knowledge_base import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma indexes for Part III compositional-agent synthesis.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--which",
        nargs="+",
        default=["part_i_questions", "part_ii_questions", "tools"],
        choices=["part_i_questions", "part_ii_questions", "tools"],
    )
    args = parser.parse_args()

    config = ProjectConfig(dataset_root=args.dataset_root, device=args.device)
    for kind in args.which:
        db = build_index(kind, config, recreate=args.recreate)
        print(f"[OK] {kind}: {db.count()} documents")


if __name__ == "__main__":
    main()
