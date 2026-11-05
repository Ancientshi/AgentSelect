#!/usr/bin/env python3
"""Convert the released JSONL dataset to the layout used by AgentSelect baselines."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

PARTS = ("PartI", "PartII", "PartIII")


def records(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def keyed(path, field):
    result = {}
    for record in records(path):
        key = record.get(field)
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}: missing or invalid {field}")
        if key in result:
            raise ValueError(f"{path}: duplicate {field}: {key}")
        result[key] = record
    return result


def query_id(part, source_id):
    payload = json.dumps({"part": part, "source_query_id": source_id},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "q_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))


def convert(source, dest):
    seen_qids, seen_agents = {}, {}
    summaries = []
    for part in PARTS:
        root = source / part
        queries = keyed(root / "queries.jsonl", "query_id")
        agents = keyed(root / "agents.jsonl", "agent_id")
        rankings = keyed(root / "rankings.jsonl", "query_id")
        converted_agents = {}
        for aid, record in agents.items():
            llm, tools = record.get("llm"), record.get("tools")
            if llm is not None and not isinstance(llm, str):
                raise ValueError(f"{part}: invalid llm for {aid}")
            if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
                raise ValueError(f"{part}: invalid tools for {aid}")
            agent = {"M": {"id": llm or "", "name": llm or ""}, "T": {"tools": tools}}
            if aid in seen_agents and seen_agents[aid] != agent:
                raise ValueError(f"Conflicting agent ID across parts: {aid}")
            seen_agents[aid] = agent
            converted_agents[aid] = agent

        converted_questions, converted_rankings = {}, {}
        for source_id, record in queries.items():
            text = record.get("query")
            if not isinstance(text, str):
                raise ValueError(f"{part}: invalid query text for {source_id}")
            qid = query_id(part, source_id)
            if qid in seen_qids:
                raise ValueError(f"Query ID collision: {part}/{source_id} and {seen_qids[qid]}")
            seen_qids[qid] = f"{part}/{source_id}"
            converted_questions[qid] = {"input": text, "source_query_id": source_id, "part": part}
        for source_id, record in rankings.items():
            if source_id not in queries:
                raise ValueError(f"{part}: ranking refers to missing query {source_id}")
            candidates = record.get("agent_ids")
            if not isinstance(candidates, list) or any(not isinstance(a, str) or a not in agents for a in candidates):
                raise ValueError(f"{part}: invalid or missing agent reference in ranking {source_id}")
            converted_rankings[query_id(part, source_id)] = candidates
        write_json(dest / part / "agents" / "merge.json", converted_agents)
        write_json(dest / part / "questions" / "merge.json", converted_questions)
        write_json(dest / part / "rankings" / "merge.json", {"rankings": converted_rankings})
        summaries.append(f"{part}: {len(queries):,} queries, {len(agents):,} agents, {len(rankings):,} rankings")

    for name, output in (("tools", "Tools"), ("llms", "LLMs")):
        items = keyed(source / f"{name}.jsonl", "name")
        write_json(dest / output / "merge.json", items)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Downloaded JSONL dataset directory")
    parser.add_argument("--output", required=True, type=Path, help="New directory for converted data")
    args = parser.parse_args()
    source, dest = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error(f"input directory does not exist: {source}")
    if dest.exists():
        parser.error(f"output already exists; choose a new directory: {dest}")
    if source == dest or source in dest.parents or dest in source.parents:
        parser.error("input and output directories must not contain one another")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        summaries = convert(source, temp)
        os.rename(temp, dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print("\n".join(summaries))
    print(f"Prepared dataset: {dest}")


if __name__ == "__main__":
    main()
