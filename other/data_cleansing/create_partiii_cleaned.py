#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter, defaultdict


DEFAULT_LLM = "Default_LLM"
DEFAULT_TOOL = "Default_Tool"

# Input paths
SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_AGENTS_PATH = SCRIPT_DIR / "merge.cleaned.json"
RAW_RANKINGS_PATH = Path("../dataset/PartIII/rankings/merge.json")

# Output paths
OUTPUT_ROOT = SCRIPT_DIR / "PartIII_cleaned"
OUTPUT_AGENTS_DIR = OUTPUT_ROOT / "agents"
OUTPUT_RANKINGS_DIR = OUTPUT_ROOT / "rankings"

OUTPUT_AGENTS_PATH = OUTPUT_AGENTS_DIR / "merge.json"
OUTPUT_RANKINGS_PATH = OUTPUT_RANKINGS_DIR / "merge.json"
OUTPUT_SUMMARY_PATH = OUTPUT_ROOT / "cleaning_summary.json"

OUTPUT_AGENTS_PATH = OUTPUT_AGENTS_DIR / "merge.json"
OUTPUT_RANKINGS_PATH = OUTPUT_RANKINGS_DIR / "merge.json"
OUTPUT_SUMMARY_PATH = OUTPUT_ROOT / "cleaning_summary.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_llm_name(agent_info: dict) -> str:
    if not isinstance(agent_info, dict):
        return ""

    m = agent_info.get("M", {})
    if not isinstance(m, dict):
        return ""

    return str(m.get("name", "")).strip()


def get_tools(agent_info: dict) -> list:
    if not isinstance(agent_info, dict):
        return []

    t = agent_info.get("T", {})

    if isinstance(t, dict):
        tools = t.get("tools", [])
    elif isinstance(t, list):
        tools = t
    else:
        tools = []

    if isinstance(tools, str):
        tools = [tools]

    if not isinstance(tools, list):
        return []

    return [str(x).strip() for x in tools]


def is_valid_agent(agent_info: dict) -> bool:
    """
    A valid PartIII agent must not use Default_LLM or Default_Tool.
    Empty tool list is allowed, because some agents may be LLM-only.
    """
    llm_name = get_llm_name(agent_info)
    tools = get_tools(agent_info)

    if not llm_name:
        return False

    if llm_name == DEFAULT_LLM:
        return False

    if DEFAULT_TOOL in tools:
        return False

    return True


def main():
    cleaned_agents = load_json(CLEANED_AGENTS_PATH)
    ranking_data = load_json(RAW_RANKINGS_PATH)

    rankings = ranking_data.get("rankings", {})
    if not isinstance(rankings, dict):
        raise ValueError("Invalid rankings format: expected ranking_data['rankings'] to be a dict.")

    valid_agents = {}
    invalid_agents = {}

    invalid_reason_counter = Counter()

    for agent_id, agent_info in cleaned_agents.items():
        llm_name = get_llm_name(agent_info)
        tools = get_tools(agent_info)

        reasons = []

        if not llm_name:
            reasons.append("missing_llm")

        if llm_name == DEFAULT_LLM:
            reasons.append("default_llm")

        if DEFAULT_TOOL in tools:
            reasons.append("default_tool")

        if reasons:
            invalid_agents[agent_id] = {
                "reasons": reasons,
                "llm": llm_name,
                "tools": tools,
            }
            for reason in reasons:
                invalid_reason_counter[reason] += 1
        else:
            valid_agents[agent_id] = agent_info

    cleaned_rankings = {}
    removed_pairs = []

    stats = defaultdict(int)
    stats["original_agent_count"] = len(cleaned_agents)
    stats["valid_agent_count"] = len(valid_agents)
    stats["invalid_agent_count"] = len(invalid_agents)
    stats["original_question_count"] = len(rankings)

    removed_agent_counter = Counter()
    original_ranking_length_counter = Counter()
    cleaned_ranking_length_counter = Counter()

    for question_id, agent_list in rankings.items():
        if not isinstance(agent_list, list):
            stats["invalid_ranking_records"] += 1
            cleaned_rankings[question_id] = []
            continue

        original_ranking_length_counter[len(agent_list)] += 1

        new_agent_list = []

        for agent_id in agent_list:
            if agent_id not in cleaned_agents:
                stats["removed_pairs_missing_agent"] += 1
                removed_agent_counter[agent_id] += 1
                removed_pairs.append(
                    {
                        "question_id": question_id,
                        "agent_id": agent_id,
                        "reason": "missing_agent",
                    }
                )
                continue

            if agent_id in invalid_agents:
                stats["removed_pairs_invalid_agent"] += 1
                removed_agent_counter[agent_id] += 1
                removed_pairs.append(
                    {
                        "question_id": question_id,
                        "agent_id": agent_id,
                        "reason": invalid_agents[agent_id]["reasons"],
                    }
                )
                continue

            new_agent_list.append(agent_id)

        cleaned_rankings[question_id] = new_agent_list
        cleaned_ranking_length_counter[len(new_agent_list)] += 1

        stats["original_pair_count"] += len(agent_list)
        stats["cleaned_pair_count"] += len(new_agent_list)

        if len(new_agent_list) == 0:
            stats["questions_with_empty_ranking_after_cleaning"] += 1

        if len(new_agent_list) < len(agent_list):
            stats["questions_affected_by_cleaning"] += 1

    cleaned_ranking_data = dict(ranking_data)
    cleaned_ranking_data["task"] = "PartIII_cleaned"
    cleaned_ranking_data["question_count"] = len(cleaned_rankings)
    cleaned_ranking_data["rankings"] = cleaned_rankings

    summary = {
        "input_files": {
            "cleaned_agents": str(CLEANED_AGENTS_PATH),
            "raw_rankings": str(RAW_RANKINGS_PATH),
        },
        "output_files": {
            "cleaned_agents": str(OUTPUT_AGENTS_PATH),
            "cleaned_rankings": str(OUTPUT_RANKINGS_PATH),
            "summary": str(OUTPUT_SUMMARY_PATH),
        },
        "statistics": dict(stats),
        "invalid_agent_reason_count": dict(invalid_reason_counter.most_common()),
        "original_ranking_length_distribution": {
            str(k): v for k, v in sorted(original_ranking_length_counter.items())
        },
        "cleaned_ranking_length_distribution": {
            str(k): v for k, v in sorted(cleaned_ranking_length_counter.items())
        },
        "top_removed_agents": dict(removed_agent_counter.most_common(50)),
        "invalid_agents": invalid_agents,
        "removed_pairs_preview": removed_pairs[:100],
    }

    save_json(valid_agents, OUTPUT_AGENTS_PATH)
    save_json(cleaned_ranking_data, OUTPUT_RANKINGS_PATH)
    save_json(summary, OUTPUT_SUMMARY_PATH)

    print("=" * 80)
    print("PartIII_cleaned dataset created.")
    print("=" * 80)

    print(f"Cleaned agents input: {CLEANED_AGENTS_PATH}")
    print(f"Raw rankings input: {RAW_RANKINGS_PATH}")
    print()

    print("[Agent statistics]")
    print(f"Original agents: {stats['original_agent_count']}")
    print(f"Valid agents: {stats['valid_agent_count']}")
    print(f"Invalid agents removed from agent pool: {stats['invalid_agent_count']}")
    print()

    print("[Ranking statistics]")
    print(f"Questions: {stats['original_question_count']}")
    print(f"Original question-agent pairs: {stats['original_pair_count']}")
    print(f"Cleaned question-agent pairs: {stats['cleaned_pair_count']}")
    print(f"Removed invalid pairs: {stats['removed_pairs_invalid_agent']}")
    print(f"Removed missing-agent pairs: {stats['removed_pairs_missing_agent']}")
    print(f"Questions affected by cleaning: {stats['questions_affected_by_cleaning']}")
    print(f"Questions with empty ranking after cleaning: {stats['questions_with_empty_ranking_after_cleaning']}")
    print()

    print("[Invalid agent reasons]")
    for reason, count in invalid_reason_counter.most_common():
        print(f"{count:>8}  {reason}")
    print()

    print("[Cleaned ranking length distribution]")
    for length, count in sorted(cleaned_ranking_length_counter.items()):
        print(f"length={length}: {count}")
    print()

    print("[Saved files]")
    print(f"Agents: {OUTPUT_AGENTS_PATH}")
    print(f"Rankings: {OUTPUT_RANKINGS_PATH}")
    print(f"Summary: {OUTPUT_SUMMARY_PATH}")

# python create_partiii_cleaned.py
if __name__ == "__main__":
    main()