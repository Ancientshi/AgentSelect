#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for cross-encoder / external-service reranking evaluation.

The helpers here mirror the standalone BGE/EasyRec scripts the user provided,
but are adapted to the repository's data structures and constants so they can
be reused across different evaluators.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from agent_rec.config import pos_topk_for_qid
from agent_rec.data import stratified_train_valid_split
from agent_rec.eval import sample_eval_candidates


@dataclass
class EvalItem:
    qid: str
    qtext: str
    cand_ids: List[str]
    doc_texts: List[str]
    rel_set: Set[str]


def build_agent_text_cache(all_agents: Dict[str, dict], tools: Dict[str, dict]) -> Dict[str, str]:
    """
    Build a textual representation for each agent: `<model> || tool desc`.

    The same concatenation is used by the original BGE/EasyRec scripts, so
    keeping it here ensures score parity when migrating to this codebase.
    """

    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {}) or {}
        return f"{tn} {t.get('description', '')}".strip()

    cache: Dict[str, str] = {}
    for aid, a in all_agents.items():
        a = a or {}
        mname = (a.get("M", {}) or {}).get("name", "") or ""
        tlst = (a.get("T", {}) or {}).get("tools", []) or []
        parts = [mname]
        if tlst:
            tool_parts = [_tool_text(tn) for tn in tlst]
            parts.append(" || " + " | ".join(tool_parts))
        cache[aid] = "".join(parts).strip(" |")
    return cache


def select_eval_qids(
    qids_in_rank: List[str],
    *,
    seed: int,
    valid_ratio: float = 0.2,
    qid_to_part: Dict[str, str] | None = None,
    stratified: bool = True,
) -> List[str]:
    """
    Pick eval qids in a deterministic way.

    If `qid_to_part` is provided (default for our datasets) and `stratified` is
    True, this mirrors the training scripts' part-aware split so evaluation
    uses the same distribution as the traditional models. Otherwise, it falls
    back to a simple global shuffle.
    """
    if valid_ratio <= 0:
        return list(qids_in_rank)
    if stratified and qid_to_part:
        _, valid_qids = stratified_train_valid_split(
            list(qids_in_rank), qid_to_part=qid_to_part, valid_ratio=valid_ratio, seed=seed
        )
        return valid_qids

    rng = random.Random(seed)
    eval_qids = list(qids_in_rank)
    rng.shuffle(eval_qids)
    n_valid = int(len(eval_qids) * valid_ratio)
    return eval_qids[:n_valid]


def sample_qids_by_part(
    qids: Sequence[str],
    *,
    qid_to_part: Dict[str, str] | None,
    per_part: int,
    seed: int,
) -> List[str]:
    """Sample a fixed number of qids from each dataset part.

    Args:
        qids: Candidate qids to sample from.
        qid_to_part: Mapping from qid to part name.
        per_part: Number of qids to sample per part; 0 or negative disables sampling.
        seed: Seed for deterministic sampling.
    """

    if per_part <= 0 or not qid_to_part:
        return list(qids)

    rng = random.Random(seed)
    by_part: Dict[str, List[str]] = defaultdict(list)
    for qid in qids:
        part = qid_to_part.get(qid, "unknown")
        by_part[part].append(qid)

    sampled: List[str] = []
    for part, part_qids in by_part.items():
        if len(part_qids) <= per_part:
            sampled.extend(part_qids)
        else:
            sampled.extend(rng.sample(part_qids, per_part))

    return sampled


def prepare_eval_items(
    *,
    eval_qids: Iterable[str],
    all_questions: Dict[str, dict],
    all_agents: Dict[str, dict],
    tools: Dict[str, dict],
    all_rankings: Dict[str, List[str]],
    a_ids: List[str],
    seed: int,
    cand_size: int = 1000,
    pos_topk: int | None = None,
    qid_to_part: Dict[str, str] | None = None,
    agent_text_cache: Dict[str, str] | None = None,
) -> List[EvalItem]:
    """
    Construct evaluation items with positives injected and random negatives.
    """
    if cand_size <= 0:
        raise ValueError("Candidate size must be positive")
    a_set = set(a_ids)
    catalog = sorted(a_set)

    if agent_text_cache is None:
        agent_text_cache = build_agent_text_cache(all_agents, tools)

    items: List[EvalItem] = []
    for qid in eval_qids:
        k = pos_topk if pos_topk is not None else pos_topk_for_qid(qid, qid_to_part)
        gt = [aid for aid in (all_rankings.get(qid) or [])[:k] if aid in a_set]
        if not gt:
            continue
        rel_set = set(gt)

        cand_ids = sample_eval_candidates(catalog, gt, cand_size, qid=qid, seed=seed)

        qtext = (all_questions.get(qid, {}) or {}).get("input", "") or ""
        doc_texts = [agent_text_cache.get(aid, "") for aid in cand_ids]

        items.append(
            EvalItem(
                qid=qid,
                qtext=qtext,
                cand_ids=cand_ids,
                doc_texts=doc_texts,
                rel_set=rel_set,
            )
        )

    return items


def metric_template(ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    return {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}


def metrics_from_hits(bin_hits: List[int], rel_size: int, ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    rel_size = max(rel_size, 1)
    for k in ks:
        topk_hits = bin_hits[:k]
        Hk = sum(topk_hits)
        P = Hk / float(k)
        R = Hk / float(rel_size)
        F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
        Hit = 1.0 if Hk > 0 else 0.0

        dcg = 0.0
        for i, h in enumerate(topk_hits):
            if h:
                dcg += 1.0 / math.log2(i + 2.0)
        ideal = min(rel_size, k)
        idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
        nDCG = (dcg / idcg) if idcg > 0 else 0.0

        rr = 0.0
        for i, h in enumerate(topk_hits):
            if h:
                rr = 1.0 / float(i + 1)
                break

        out[k] = {"P": P, "R": R, "F1": F1, "Hit": Hit, "nDCG": nDCG, "MRR": rr}
    return out


def accumulate_metrics(
    agg: Dict[int, Dict[str, float]], metrics: Dict[int, Dict[str, float]], ks: Sequence[int]
) -> None:
    for k in ks:
        for m, v in metrics[k].items():
            agg[k][m] += v


def finalize_metrics(agg: Dict[int, Dict[str, float]], count: int, ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    if count == 0:
        return metric_template(ks)
    out = metric_template(ks)
    for k in ks:
        for m in agg[k]:
            out[k][m] = agg[k][m] / float(count)
    return out


def topk_hits_from_scores(
    scores: np.ndarray, cand_ids: List[str], rel_set: Set[str], ks: Sequence[int]
) -> Tuple[List[str], List[int]]:
    if scores.size == 0 or len(cand_ids) == 0:
        return [], []
    max_k = max(ks)
    order = np.argsort(-scores, kind="stable")[:max_k]
    pred_ids = [cand_ids[i] for i in order]
    bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
    return pred_ids, bin_hits
