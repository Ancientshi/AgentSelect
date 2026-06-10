#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from partiii_compositional_agents.config import ProjectConfig
from partiii_compositional_agents.synthesis import synthesize_from_part_ii


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Part III compositional agents from Part II queries.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("./outputs"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--llm-model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    config = ProjectConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        device=args.device,
        llm_model=args.llm_model,
        max_workers=args.max_workers,
    )
    out_path = synthesize_from_part_ii(config, sample_size=args.sample_size)
    print(f"[OK] results written to {out_path}")


if __name__ == "__main__":
    main()
