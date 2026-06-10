from __future__ import annotations

import json
import random
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import ProjectConfig
from .io_utils import append_jsonl, load_json
from .knowledge_base import KnowledgeBase, VectorDatabase, build_bge_embedding
from .llm_utils import generate_agents_for_question, generate_tool_search_queries
from .search import Search


def format_prompt_from_agents(ordered_agents: list[tuple[str, dict[str, Any]]], title: str) -> str:
    lines = [title, "The following candidate agents are ranked from highest priority to lowest:"]
    for i, (agent_id, info) in enumerate(ordered_agents, start=1):
        lines.append(f"{i}. {agent_id}: {json.dumps(info, ensure_ascii=False)}")
    return "\n".join(lines)


def tools_search_prompt(tool_search_engine: Search, query: str, threshold: float, model_name: str) -> str:
    prompt_parts: list[str] = []
    for tool_query in generate_tool_search_queries(query, model_name=model_name):
        docs = tool_search_engine.search(tool_query)
        body = "".join(
            f"{doc.metadata.get('tool_name', 'UNKNOWN_TOOL')}: {doc.page_content}\n"
            for doc in docs
        )
        prompt_parts.append(
            f"#### Relevant tools (score threshold >= {threshold}) about '{tool_query}':\n{body}"
        )
    return "\n\n".join(prompt_parts)


def _prepare_search_engines(config: ProjectConfig, question_index: Path) -> tuple[Search, Search]:
    embeddings = build_bge_embedding(config)
    question_db = VectorDatabase(embeddings, question_index)
    tool_db = VectorDatabase(embeddings, config.tool_vector_db)
    return (
        Search(KnowledgeBase(question_db), score_threshold=config.questions_score_threshold, k=config.questions_topk),
        Search(KnowledgeBase(tool_db), score_threshold=config.tools_score_threshold, k=config.tools_topk),
    )


def synthesize_from_part_i(config: ProjectConfig, train_list_filename: str, sample_size: int | None = 5) -> Path:
    """Generate compositional agents for Part I-style queries.

    The suggestion combines directly ranked Part I backbone agents, retrieved Part II toolkit agents,
    and retrieved tool descriptions.
    """
    random.seed(config.seed)
    questions_search_engine, tool_search_engine = _prepare_search_engines(config, config.part_ii_vector_db)

    part_i_rankings = load_json(config.part_i_dir / "rankings" / "merge.json")
    part_i_agents = load_json(config.part_i_dir / "agents" / "merge.json")
    part_i_questions = load_json(config.part_i_dir / "questions" / "merge.json")
    part_ii_rankings = load_json(config.part_ii_dir / "rankings" / "merge.json")
    part_ii_agents = load_json(config.part_ii_dir / "agents" / "merge.json")

    def process_one(dirname: str, num: int) -> dict[str, Any] | None:
        try:
            key = f"leaderboard_{dirname}_question_{num}"
            query = part_i_questions.get(key, {}).get("input", "")

            direct_ranked = part_i_rankings["rankings"].get(key, [])[:5]
            prompt_i = format_prompt_from_agents(
                [(aid, part_i_agents[aid]) for aid in direct_ranked if aid in part_i_agents],
                title=f'#### Ranked recommendation agents (Backbone LLM Driven) for answering: "{query}"',
            )

            retrieved_docs = questions_search_engine.search(query)
            retrieved_agent_ids: list[str] = []
            for doc in retrieved_docs:
                meta = doc.metadata or {}
                candidate_keys = [meta.get("key"), meta.get("str_index"), f"question_{meta.get('str_index')}"]
                for candidate_key in candidate_keys:
                    if candidate_key in part_ii_rankings.get("rankings", {}):
                        retrieved_agent_ids.extend(part_ii_rankings["rankings"][candidate_key])
                        break
            retrieved_agent_ids = list(OrderedDict.fromkeys(retrieved_agent_ids))[:5]
            prompt_ii = format_prompt_from_agents(
                [(aid, part_ii_agents.get(aid, {})) for aid in retrieved_agent_ids],
                title=f"#### Ranked recommendation agents (Toolkit Driven) from similar queries (score threshold >= {config.questions_score_threshold}):",
            )

            prompt_tools = tools_search_prompt(tool_search_engine, query, config.tools_score_threshold, config.llm_model)
            suggestion = f"{prompt_i}\n\n{prompt_ii}\n\n{prompt_tools}"
            recommendation_agents, prompt = generate_agents_for_question(query, suggestion, model_name=config.llm_model)
            return {
                "dirname": dirname,
                "num": num,
                "key": key,
                "question": query,
                "suggestion": suggestion,
                "prompt": prompt,
                "recommendation_agents": recommendation_agents,
            }
        except Exception:
            traceback.print_exc()
            return None

    rows: list[dict[str, Any]] = []
    for dirname in sorted(p.name for p in Path.cwd().iterdir() if p.is_dir()):
        train_path = Path(dirname) / train_list_filename
        if not train_path.exists():
            continue
        nums = load_json(train_path)
        if sample_size is not None and len(nums) > sample_size:
            nums = sorted(random.sample(nums, sample_size))
        with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
            futures = {ex.submit(process_one, dirname, int(num)): num for num in nums}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"processing {dirname}"):
                row = fut.result()
                if row is not None:
                    rows.append(row)

    out_path = config.output_root / "simulated_results_from_PartI_rebuttal" / "main.jsonl"
    append_jsonl(out_path, rows)
    return out_path


def synthesize_from_part_ii(config: ProjectConfig, sample_size: int | None = 200) -> Path:
    """Generate compositional agents for Part II-style queries.

    The suggestion combines directly ranked Part II toolkit agents, retrieved Part I backbone agents,
    and retrieved tool descriptions.
    """
    np.random.seed(config.seed)
    questions_search_engine, tool_search_engine = _prepare_search_engines(config, config.part_i_vector_db)

    part_i_rankings = load_json(config.part_i_dir / "rankings" / "merge.json")
    part_i_agents = load_json(config.part_i_dir / "agents" / "merge.json")
    part_i_questions = load_json(config.part_i_dir / "questions" / "merge.json")
    part_i_keys = list(part_i_questions.keys())
    part_ii_questions = load_json(config.part_ii_dir / "questions" / "merge.json")
    part_ii_rankings = load_json(config.part_ii_dir / "rankings" / "merge.json")
    part_ii_agents = load_json(config.part_ii_dir / "agents" / "merge.json")

    nums = list(range(len(part_ii_questions)))
    if sample_size is not None and len(nums) > sample_size:
        nums = sorted(np.random.choice(nums, size=sample_size, replace=False).astype(int).tolist())

    def process_one(num: int) -> dict[str, Any] | None:
        try:
            key = f"PartII_question_{num}"
            query = part_ii_questions.get(key, {}).get("input", "")
            direct_ranked = part_ii_rankings["rankings"].get(key, [])[:5]
            prompt_ii = format_prompt_from_agents(
                [(aid, part_ii_agents[aid]) for aid in direct_ranked if aid in part_ii_agents],
                title=f'#### Ranked recommendation agents (Toolkit Driven) for answering: "{query}"',
            )

            retrieved_docs = questions_search_engine.search(query)
            retrieved_agent_ids: list[str] = []
            for doc in retrieved_docs:
                idx = doc.metadata.get("str_index")
                if idx is None:
                    continue
                retrieved_key = part_i_keys[int(idx)]
                retrieved_agent_ids.extend(part_i_rankings["rankings"].get(retrieved_key, []))
            retrieved_agent_ids = list(OrderedDict.fromkeys(retrieved_agent_ids))[:5]
            prompt_i = format_prompt_from_agents(
                [(aid, part_i_agents.get(aid, {})) for aid in retrieved_agent_ids],
                title=f"#### Ranked recommendation agents (Backbone LLM Driven) from similar queries (score threshold >= {config.questions_score_threshold}):",
            )

            prompt_tools = tools_search_prompt(tool_search_engine, query, config.tools_score_threshold, config.llm_model)
            suggestion = f"{prompt_ii}\n\n{prompt_i}\n\n{prompt_tools}"
            recommendation_agents, prompt = generate_agents_for_question(query, suggestion, model_name=config.llm_model)
            return {
                "dirname": "PartII",
                "num": num,
                "key": key,
                "question": query,
                "suggestion": suggestion,
                "prompt": prompt,
                "recommendation_agents": recommendation_agents,
            }
        except Exception:
            traceback.print_exc()
            return None

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
        futures = {ex.submit(process_one, int(num)): num for num in nums}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="processing PartII"):
            row = fut.result()
            if row is not None:
                rows.append(row)

    out_path = config.output_root / "simulated_results_from_PartII_rebuttal" / "main.jsonl"
    append_jsonl(out_path, rows)
    return out_path
