#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter, defaultdict


DEFAULT_INPUT_PATH = Path("../dataset/PartIII/agents/merge.json")
OUTPUT_DIR = Path(__file__).resolve().parent

DEFAULT_LLM = "Default_LLM"
DEFAULT_TOOL = "Default_Tool"


TOOL_PART_NOISE_PATTERN = re.compile(
    r"^\s*Part(?:I{1,3}|I1|II1|III1|1|2|3)(?:\b|_|-|\s|\()",
    flags=re.IGNORECASE,
)


def is_noisy_llm_name(name: str) -> bool:
    """
    Coarse LLM noise detection.

    For LLM names, any name containing 'part' is treated as noise.
    This is intentional because noisy LLM names may look like:
    - PartI1_agent_70 (google__umt5-base)
    - PartIII_agent_xxx
    """
    if not isinstance(name, str):
        return True

    name_strip = name.strip()
    if not name_strip:
        return True

    return "part" in name_strip.lower()


def is_noisy_tool_name(name: str) -> bool:
    """
    Strict tool noise detection.

    Only filters tool names that start with Part-style noisy prefixes.
    This avoids removing valid tools that merely contain 'part'.
    """
    if not isinstance(name, str):
        return True

    name_strip = name.strip()
    if not name_strip:
        return True

    return bool(TOOL_PART_NOISE_PATTERN.match(name_strip))


def normalize_frequency(counter: Counter) -> dict:
    total = sum(counter.values())
    if total == 0:
        return {}

    return {
        key: value / total
        for key, value in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    }


def extract_llm(agent_info: dict) -> str:
    m = agent_info.get("M", {})

    if isinstance(m, dict):
        return m.get("name", "")

    return ""


def extract_tools(agent_info: dict) -> list:
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
        tools = []

    return tools


def main():
    input_path = DEFAULT_INPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        agents = json.load(f)

    llm_counter = Counter()
    tool_counter = Counter()

    noisy_llm_counter = Counter()
    noisy_tool_counter = Counter()

    cleaned_agents = {}
    stats = defaultdict(int)

    for agent_id, agent_info in agents.items():
        stats["num_agents"] += 1

        if not isinstance(agent_info, dict):
            stats["invalid_agent_records"] += 1
            continue

        # ---------- LLM cleansing ----------
        raw_llm = extract_llm(agent_info)

        if is_noisy_llm_name(raw_llm):
            cleaned_llm = DEFAULT_LLM
            noisy_llm_counter[str(raw_llm)] += 1
            stats["num_noisy_llm_occurrences"] += 1
        else:
            cleaned_llm = raw_llm.strip()

        llm_counter[cleaned_llm] += 1

        # ---------- Tool cleansing ----------
        raw_tools = extract_tools(agent_info)
        cleaned_tools = []

        if len(raw_tools) == 0:
            stats["agents_without_tools"] += 1

        for tool in raw_tools:
            if is_noisy_tool_name(tool):
                cleaned_tool = DEFAULT_TOOL
                noisy_tool_counter[str(tool)] += 1
                stats["num_noisy_tool_occurrences"] += 1
            else:
                cleaned_tool = str(tool).strip()

            cleaned_tools.append(cleaned_tool)
            tool_counter[cleaned_tool] += 1
            stats["num_tool_occurrences"] += 1

        # ---------- Save cleaned agent record ----------
        cleaned_record = dict(agent_info)

        original_m = cleaned_record.get("M", {})
        original_t = cleaned_record.get("T", {})

        cleaned_record["M"] = dict(original_m) if isinstance(original_m, dict) else {}
        cleaned_record["M"]["name"] = cleaned_llm

        cleaned_record["T"] = dict(original_t) if isinstance(original_t, dict) else {}
        cleaned_record["T"]["tools"] = cleaned_tools

        cleaned_agents[agent_id] = cleaned_record

    llm_frequency = normalize_frequency(llm_counter)
    tool_frequency = normalize_frequency(tool_counter)

    summary = {
        "input_path": str(input_path),
        "num_agents": stats["num_agents"],
        "invalid_agent_records": stats["invalid_agent_records"],
        "num_unique_llms_after_cleaning": len(llm_counter),
        "num_unique_tools_after_cleaning": len(tool_counter),
        "num_tool_occurrences": stats["num_tool_occurrences"],
        "agents_without_tools": stats["agents_without_tools"],
        "num_noisy_llm_occurrences": stats["num_noisy_llm_occurrences"],
        "num_noisy_tool_occurrences": stats["num_noisy_tool_occurrences"],
        "default_llm_count": llm_counter.get(DEFAULT_LLM, 0),
        "default_tool_count": tool_counter.get(DEFAULT_TOOL, 0),
        "noisy_llm_items": dict(noisy_llm_counter.most_common()),
        "noisy_tool_items": dict(noisy_tool_counter.most_common()),
        "top_20_llms": dict(llm_counter.most_common(20)),
        "top_20_tools": dict(tool_counter.most_common(20)),
    }

    outputs = {
        "cleaned_agents": OUTPUT_DIR / "merge.cleaned.json",
        "llm_frequency": OUTPUT_DIR / "llm_frequency.json",
        "tool_frequency": OUTPUT_DIR / "tool_frequency.json",
        "llm_count": OUTPUT_DIR / "llm_count.json",
        "tool_count": OUTPUT_DIR / "tool_count.json",
        "summary": OUTPUT_DIR / "frequency_cleansing_summary.json",
    }

    with open(outputs["cleaned_agents"], "w", encoding="utf-8") as f:
        json.dump(cleaned_agents, f, ensure_ascii=False, indent=2)

    with open(outputs["llm_frequency"], "w", encoding="utf-8") as f:
        json.dump(llm_frequency, f, ensure_ascii=False, indent=2)

    with open(outputs["tool_frequency"], "w", encoding="utf-8") as f:
        json.dump(tool_frequency, f, ensure_ascii=False, indent=2)

    with open(outputs["llm_count"], "w", encoding="utf-8") as f:
        json.dump(dict(llm_counter.most_common()), f, ensure_ascii=False, indent=2)

    with open(outputs["tool_count"], "w", encoding="utf-8") as f:
        json.dump(dict(tool_counter.most_common()), f, ensure_ascii=False, indent=2)

    with open(outputs["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Data cleansing finished.")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    print("[Basic statistics]")
    print(f"Number of agents: {summary['num_agents']}")
    print(f"Invalid agent records: {summary['invalid_agent_records']}")
    print(f"Number of unique LLMs after cleaning: {summary['num_unique_llms_after_cleaning']}")
    print(f"Number of unique tools after cleaning: {summary['num_unique_tools_after_cleaning']}")
    print(f"Number of tool occurrences: {summary['num_tool_occurrences']}")
    print(f"Agents without tools: {summary['agents_without_tools']}")
    print()

    print("[Noise statistics]")
    print(f"Noisy LLM occurrences: {summary['num_noisy_llm_occurrences']}")
    print(f"Noisy tool occurrences: {summary['num_noisy_tool_occurrences']}")
    print(f"{DEFAULT_LLM} count: {summary['default_llm_count']}")
    print(f"{DEFAULT_TOOL} count: {summary['default_tool_count']}")
    print()

    print("[Noisy LLM items]")
    if noisy_llm_counter:
        for name, count in noisy_llm_counter.most_common(30):
            print(f"{count:>6}  {name}")
    else:
        print("No noisy LLM items found.")
    print()

    print("[Noisy tool items]")
    if noisy_tool_counter:
        for name, count in noisy_tool_counter.most_common(30):
            print(f"{count:>6}  {name}")
    else:
        print("No noisy tool items found.")
    print()

    print("[Top 20 LLMs]")
    for name, count in llm_counter.most_common(20):
        print(f"{count:>6}  {name}")
    print()

    print("[Top 20 tools]")
    for name, count in tool_counter.most_common(20):
        print(f"{count:>6}  {name}")
    print()

    print("[Saved files]")
    for key, path in outputs.items():
        print(f"{key}: {path}")

# python clean_agent_frequency.py
if __name__ == "__main__":
    main()