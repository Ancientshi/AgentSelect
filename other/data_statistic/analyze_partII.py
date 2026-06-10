import argparse
import json
import math
from collections import Counter
from glob import glob
from pathlib import Path
from statistics import mean, median

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None


def remove_outliers_iqr(values, iqr_mult=1.5, max_percentile=0.99):
    """
    Apply a fixed validity filter for the number of tools per agent.
    Only agents with no more than 10 tools are retained.

    The function signature is kept for compatibility with previous calls.
    """
    if not values:
        return values, {"lower": 0.0, "upper": 10.0, "removed": 0}

    arr = np.array(values, dtype=float)
    lower, upper = 0.0, 10.0
    mask = (arr >= lower) & (arr <= upper)

    cleaned = arr[mask].astype(int).tolist()
    info = {
        "lower": lower,
        "upper": upper,
        "removed": int((~mask).sum()),
    }
    return cleaned, info


def pick_json(path: Path, prefer: str = "merge.json"):
    """
    Return the preferred JSON file if it exists.
    Otherwise, return the first available JSON file in the directory.
    """
    if (path / prefer).exists():
        return path / prefer

    candidates = sorted(Path(p) for p in glob(str(path / "*.json")))
    if not candidates:
        return None

    skip_names = {"merged_tools.json"}
    filtered = [p for p in candidates if p.name not in skip_names]
    return (filtered or candidates)[0]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze(root: Path, topn: int = 20):
    questions_file = pick_json(root / "questions")
    agents_file = pick_json(root / "agents")
    rankings_file = pick_json(root / "rankings")

    tools_file = root / "tools" / "merged_tools.json"
    if not tools_file.exists():
        tools_file = pick_json(root / "tools")

    if not all([questions_file, agents_file, rankings_file]):
        raise SystemExit(
            f"[ERROR] Missing dataset files under {root}. "
            "Expected subdirectories: questions/, agents/, rankings/."
        )

    print(f"[INFO] Loading questions from: {questions_file}")
    questions = load_json(questions_file)

    print(f"[INFO] Loading agents from: {agents_file}")
    agents = load_json(agents_file)

    print(f"[INFO] Loading rankings from: {rankings_file}")
    rankings = load_json(rankings_file)

    tools_catalog = {}
    if tools_file:
        print(f"[INFO] Loading tools from: {tools_file}")
        tools_catalog = load_json(tools_file)

    # Basic dataset statistics
    num_questions = len(questions)
    num_agents = len(agents)

    question_to_agents = rankings.get("rankings", {})
    num_interactions = sum(len(v) for v in question_to_agents.values())

    total_possible = num_questions * num_agents if num_questions and num_agents else 0
    density = num_interactions / total_possible if total_possible else 0.0
    sparsity = 1.0 - density

    print(f"[INFO] Questions: {num_questions}")
    print(f"[INFO] Agents: {num_agents}")
    print(f"[INFO] Observed interactions: {num_interactions}")
    print(f"[INFO] Density: {density:.6f}")
    print(f"[INFO] Sparsity: {sparsity:.6f}")

    # Tool statistics
    defined_tools = set(tools_catalog.keys()) if isinstance(tools_catalog, dict) else set()

    agent_tools = {}
    all_used_tools = set()
    tools_per_agent = []

    for agent_id, agent in agents.items():
        tool_list = (agent.get("T", {}) or {}).get("tools", []) or []
        agent_tools[agent_id] = list(tool_list)
        tools_per_agent.append(len(tool_list))
        all_used_tools.update(tool_list)

    num_defined_tools = len(defined_tools)
    num_used_tools = len(all_used_tools)
    missing_tool_definitions = sorted(all_used_tools - defined_tools)

    tool_popularity = Counter()
    for tool_list in agent_tools.values():
        tool_popularity.update(set(tool_list))

    tool_count_distribution = Counter(tools_per_agent)

    cleaned_tools_per_agent, clip_info = remove_outliers_iqr(tools_per_agent)

    tool_count_stats = {
        "min": min(tools_per_agent) if tools_per_agent else 0,
        "max": max(tools_per_agent) if tools_per_agent else 0,
        "mean": round(mean(tools_per_agent), 4) if tools_per_agent else 0,
        "median": median(tools_per_agent) if tools_per_agent else 0,
        "p90": (
            sorted(tools_per_agent)[math.floor(0.9 * (len(tools_per_agent) - 1))]
            if tools_per_agent
            else 0
        ),
    }

    script_dir = Path(__file__).resolve().parent
    output_dir = ensure_dir(script_dir / "analysis")

    # Figure 1: raw distribution of tools per agent
    plt.figure(figsize=(8, 5))
    plt.hist(tools_per_agent, bins="auto")
    plt.title("Distribution of Tools per Agent")
    plt.xlabel("Number of Tools per Agent")
    plt.ylabel("Number of Agents")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_raw_hist = output_dir / "tools_per_agent_hist_raw.png"
    plt.savefig(fig_raw_hist, dpi=220)
    plt.close()

    # Figure 2: filtered distribution of tools per agent
    plt.figure(figsize=(8, 5))
    plt.hist(cleaned_tools_per_agent, bins="auto")
    plt.title("Distribution of Tools per Agent after Filtering")
    plt.xlabel("Number of Tools per Agent")
    plt.ylabel("Number of Agents")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_filtered_hist = output_dir / "tools_per_agent_hist_filtered.png"
    plt.savefig(fig_filtered_hist, dpi=220)
    plt.close()

    # Figure 3: top-N tool popularity
    top_tools = tool_popularity.most_common(topn)
    labels = [name for name, _ in top_tools]
    values = [count for _, count in top_tools]

    plt.figure(figsize=(max(8, min(20, 0.4 * len(labels))), 6))
    plt.bar(range(len(labels)), values)
    plt.title(f"Tool Popularity across Agents (Top {len(labels)})")
    plt.xlabel("Tool")
    plt.ylabel("Number of Agents Using the Tool")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_tool_popularity = output_dir / "tool_popularity_topN.png"
    plt.savefig(fig_tool_popularity, dpi=220)
    plt.close()

    # Export tables
    if pd is not None:
        summary_rows = [
            {
                "num_questions": num_questions,
                "num_agents": num_agents,
                "num_interactions": num_interactions,
                "density": round(density, 6),
                "sparsity": round(sparsity, 6),
                "num_tools_defined": num_defined_tools,
                "num_tools_used_by_agents": num_used_tools,
                "tools_per_agent_min": tool_count_stats["min"],
                "tools_per_agent_max": tool_count_stats["max"],
                "tools_per_agent_mean": tool_count_stats["mean"],
                "tools_per_agent_median": tool_count_stats["median"],
                "tools_per_agent_p90": tool_count_stats["p90"],
                "filter_lower_bound": round(clip_info["lower"], 4),
                "filter_upper_bound": round(clip_info["upper"], 4),
                "num_filtered_agents": clip_info["removed"],
                "agents_file": agents_file.name,
                "questions_file": questions_file.name,
                "rankings_file": rankings_file.name,
                "tools_file": tools_file.name if tools_file else "",
                "interaction_heatmap": "skipped",
            }
        ]
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

        pd.DataFrame(
            sorted(tool_count_distribution.items()),
            columns=["tools_per_agent", "num_agents"],
        ).to_csv(output_dir / "tools_per_agent_distribution.csv", index=False)

        pd.DataFrame(
            top_tools,
            columns=["tool", "num_agents"],
        ).to_csv(output_dir / f"tool_popularity_top{len(top_tools)}.csv", index=False)

    # Markdown report
    report = []
    report.append(f"# Dataset Audit: {root.name}\n")

    report.append("## Overview\n")
    report.append(
        f"- Questions: {num_questions}\n"
        f"- Agents: {num_agents}\n"
        f"- Observed interactions: {num_interactions}\n"
        f"- Matrix size: {num_questions} × {num_agents} = {total_possible}\n"
        f"- Density: {density:.6f}\n"
        f"- Sparsity: {sparsity:.6f}\n"
    )

    report.append("## Tool Coverage\n")
    report.append(
        f"- Tools defined in the catalog: {num_defined_tools}\n"
        f"- Unique tools used by agents: {num_used_tools}\n"
    )

    if missing_tool_definitions:
        preview = ", ".join(missing_tool_definitions[:20])
        suffix = " ..." if len(missing_tool_definitions) > 20 else ""
        report.append(
            f"- Warning: {len(missing_tool_definitions)} tools used by agents "
            f"are missing from the catalog: {preview}{suffix}\n"
        )

    report.append("## Tools per Agent\n")
    report.append(
        "- Min / median / mean / p90 / max: "
        f"{tool_count_stats['min']} / "
        f"{tool_count_stats['median']} / "
        f"{tool_count_stats['mean']} / "
        f"{tool_count_stats['p90']} / "
        f"{tool_count_stats['max']}\n"
    )

    report.append("## Generated Figures\n")
    report.append(f"1. Tools per agent, raw distribution: `{fig_raw_hist.name}`\n")
    report.append(
        f"2. Tools per agent, filtered distribution: `{fig_filtered_hist.name}` "
        f"(lower={clip_info['lower']:.2f}, "
        f"upper={clip_info['upper']:.2f}, "
        f"removed={clip_info['removed']})\n"
    )
    report.append(f"3. Tool popularity, top {topn}: `{fig_tool_popularity.name}`\n")
    report.append("4. Interaction matrix heatmap: skipped to avoid excessive memory and runtime.\n")

    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"[OK] Dataset audit completed. Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit the PartII agent-selection dataset."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="../dataset/PartII",
        help="Path to the dataset root directory. Default: ../dataset/PartII",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="Number of most frequent tools to show in the popularity chart.",
    )

    args = parser.parse_args()
    analyze(Path(args.root).resolve(), topn=args.topn)


if __name__ == "__main__":
    main()