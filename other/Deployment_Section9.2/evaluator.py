#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Utils
# -----------------------------

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur

INVALID_MARKERS = [
    "tool▁calls▁begin",
    "tool_calls_begin",
    "<tool_calls_begin>",
    "tool▁calls▁end",
    "tool_calls_end",
    "<tool_calls_end>",
    "tool▁call▁begin",
    "tool_call_begin",
    "<tool_call_begin>",
]

def extract_assistant_visible_texts(run_json: dict) -> List[str]:
    msgs = safe_get(run_json, ["request", "response", "messages"], default=[]) or []
    outs: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if any(marker in content for marker in INVALID_MARKERS):
            continue
        outs.append(content)
    return outs

def try_extract_final_answer(text: str) -> str:
    t = text.strip()

    # direct JSON
    if t.startswith("{") and t.endswith("}"):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                fa = obj.get("final_answer")
                if isinstance(fa, str) and fa.strip():
                    return fa.strip()
        except Exception:
            pass

    # substring JSON
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        blob = m.group(0)
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                fa = obj.get("final_answer")
                if isinstance(fa, str) and fa.strip():
                    return fa.strip()
        except Exception:
            pass

    # regex "final_answer": "..."
    m2 = re.search(r'"final_answer"\s*:\s*"([\s\S]*?)"\s*(?:,|\})', t)
    if m2:
        s = m2.group(1)
        s = s.replace(r"\n", "\n").replace(r"\\", "\\").replace(r"\/", "/").replace(r"\"", "\"")
        return s.strip()

    return t

def normalize_whitespace(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

# -----------------------------
# True ranking inference
# -----------------------------

def infer_true_rank_key(r: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Return a sortable key where smaller is better.
    Prefer:
      1) agent_rank_pos_1indexed  (most faithful)
      2) pos_map_agent_pick_index
      3) agent_pick_index
    If none, put at end.
    """
    rank_pos = r.get("agent_rank_pos_1indexed")
    if isinstance(rank_pos, int):
        return (0, rank_pos, 0)

    pm = r.get("pos_map_agent_pick_index")
    if isinstance(pm, int):
        return (1, pm, 0)

    ap = r.get("agent_pick_index")
    if isinstance(ap, int):
        return (2, ap, 0)

    return (9, 10**9, 0)

# -----------------------------
# Metrics
# -----------------------------

def spearman_rho(true_order: List[str], pred_order: List[str]) -> Optional[float]:
    items = [x for x in true_order if x in set(pred_order)]
    n = len(items)
    if n < 2:
        return None
    tr = {a: i + 1 for i, a in enumerate(true_order)}
    pr = {a: i + 1 for i, a in enumerate(pred_order)}
    d2 = 0.0
    for a in items:
        d = tr[a] - pr[a]
        d2 += d * d
    return 1.0 - (6.0 * d2) / (n * (n * n - 1.0))

def kendall_tau(true_order: List[str], pred_order: List[str]) -> Optional[float]:
    items = [x for x in true_order if x in set(pred_order)]
    n = len(items)
    if n < 2:
        return None
    tr = {a: i for i, a in enumerate(true_order)}
    pr = {a: i for i, a in enumerate(pred_order)}
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            s1 = tr[a] - tr[b]
            s2 = pr[a] - pr[b]
            if s1 * s2 > 0:
                conc += 1
            elif s1 * s2 < 0:
                disc += 1
    denom = conc + disc
    if denom == 0:
        return None
    return (conc - disc) / denom

def ndcg_at_k(true_order: List[str], pred_order: List[str], k: int) -> Optional[float]:
    if k <= 0:
        return None

    rel = {a: (len(true_order) - i) for i, a in enumerate(true_order)}  # best gets max

    def dcg(order: List[str]) -> float:
        s = 0.0
        for idx, a in enumerate(order[:k], start=1):
            r = rel.get(a, 0)
            s += (2**r - 1) / math.log2(idx + 1)
        return s

    ideal = dcg(true_order[:k])
    if ideal == 0:
        return None
    return dcg(pred_order) / ideal

# -----------------------------
# GPT prompting & parsing
# -----------------------------

def build_rerank_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    """
    Ranking objective: task completion & tool executability first.
    We are choosing the best agent to deploy, so prefer answers that are actionable
    with the provided tools, cover all sub-requests, and avoid unjustified refusal.
    """
    lines: List[str] = []
    lines.append("You are a strict evaluator for an agent recommender benchmark.")
    lines.append("Your goal is to rank candidate answers by how well the agent can COMPLETE the user's task in practice.")
    lines.append("")
    lines.append("PRIMARY objective: Task Completion + Tool Executability (most important).")
    lines.append("")
    lines.append("Rank candidates using these dimensions IN THIS ORDER:")
    lines.append("1) Task completion: answers ALL parts of the user's request (covers every sub-question).")
    lines.append("2) Tool executability: the answer is realistically achievable using the listed tools for that candidate;")
    lines.append("   - Reward candidates whose tools are clearly relevant to the task and whose answer matches what those tools can do.")
    lines.append("   - Strongly penalize candidates that refuse or say 'I can't' when the needed info is common knowledge or achievable.")
    lines.append("   - Strongly penalize candidates that claim to have retrieved data when their tools are unrelated.")
    lines.append("3) Correctness & specificity: correct facts, correct details, minimal speculation.")
    lines.append("4) Clarity & helpfulness: well-structured, easy to follow, directly usable.")
    lines.append("5) Hallucination control: avoid fabricated URLs, fake policies, fake 'latest data retrieved', placeholder domains, etc.")
    lines.append("")
    lines.append("IMPORTANT RULES:")
    lines.append("- This is a DEPLOYMENT-centric evaluation: prefer actionable answers over safe-but-useless refusals.")
    lines.append("- If an answer provides only a small sub-part (e.g., just continent/language) but ignores the main task, rank it low.")
    lines.append("- If an answer includes detailed hotel policies / maps but cannot be supported by its tools (or looks like invented content), rank it low.")
    lines.append("- Do NOT reward generic tool-error commentary unless it directly helps complete the task.")
    lines.append("")
    lines.append("OUTPUT CONSTRAINTS:")
    lines.append("- Output MUST be valid JSON only (no markdown).")
    lines.append("- 'ranking' MUST include ALL candidate IDs exactly once, in best-to-worst order.")
    lines.append("- 'scores' MUST include a numeric score for each candidate ID (0-10).")
    lines.append("- 'reasons' MUST include 1-2 sentences per candidate, explicitly referencing task completion and tool executability.")
    lines.append("")
    lines.append(f"User query:\n{query.strip()}")
    lines.append("")
    lines.append("Candidates:")
    for c in candidates:
        cid = c["candidate_id"]
        agent_id = c.get("agent_id", "")
        llm = c.get("llm_name_oneagent", "")
        tools = c.get("tools", [])
        ans = c.get("answer_text", "")
        tools_s = ", ".join(tools[:8]) + ("..." if len(tools) > 8 else "")
        lines.append(f"\n[{cid}] agent_id={agent_id} llm={llm} tools=[{tools_s}]")
        lines.append("Answer:")
        lines.append(ans)

    lines.append("")
    lines.append("Return JSON with this schema (IDs must match the candidates above):")
    lines.append("{")
    lines.append('  "ranking": ["C1","C2","C3"],')
    lines.append('  "scores": {"C1": 8.7, "C2": 6.0, "C3": 2.5},')
    lines.append('  "reasons": {"C1": "1-2 sentences", "C2": "1-2 sentences", "C3": "1-2 sentences"}')
    lines.append("}")
    return "\n".join(lines)


def parse_json_strict(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("No JSON object found in GPT response.")
    return json.loads(m.group(0))

def _try_import_openai():
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None

# thread-local OpenAI client
_thread_local = threading.local()

def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    openai = _try_import_openai()
    if openai is None:
        raise RuntimeError("openai python package not found. Please `pip install openai`.")

    if getattr(_thread_local, "client", None) is None:
        _thread_local.client = openai.OpenAI(api_key=api_key)
    return _thread_local.client

def _gpt_chat(prompt: str, model_name: str) -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        stream=False,
        service_tier="default",
    )
    return resp.choices[0].message.content or ""

# -----------------------------
# IO: load & group by qid
# -----------------------------

def load_runs_grouped_by_qid(path: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            qid = r.get("qid")
            if not isinstance(qid, str) or not qid:
                continue
            groups.setdefault(qid, []).append(r)
    return groups

def extract_query_text(r: Dict[str, Any]) -> str:
    q = r.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q2 = safe_get(r, ["request", "payload", "query"], default="")
    if isinstance(q2, str) and q2.strip():
        return q2.strip()
    return ""

def build_candidates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for r in records:
        assistant_texts = extract_assistant_visible_texts(r)
        if not assistant_texts:
            continue

        joined = normalize_whitespace("\n\n".join(assistant_texts))
        final_ans = normalize_whitespace(try_extract_final_answer(joined))
        if not final_ans:
            continue

        c = {
            "candidate_id": f"C{len(candidates) + 1}",
            "agent_id": r.get("agent_id", ""),
            "agent_idx": r.get("agent_idx", None),
            "agent_pick_index": r.get("agent_pick_index", None),
            "pos_map_agent_pick_index": r.get("pos_map_agent_pick_index", None),
            "agent_rank_pos_1indexed": r.get("agent_rank_pos_1indexed", None),
            "llm_name_oneagent": r.get("llm_name_oneagent", ""),
            "llm_name_dataset": r.get("llm_name_dataset", ""),
            "tools": r.get("tools", []) if isinstance(r.get("tools", []), list) else [],
            "duration": safe_get(r, ["request", "response", "metrics", "duration"], default=None),
            "answer_text": final_ans,
            "_true_rank_key": infer_true_rank_key(r),
        }
        candidates.append(c)

    candidates.sort(key=lambda x: x["_true_rank_key"])
    return candidates

# -----------------------------
# Incremental save / resume
# -----------------------------

def load_done_qids(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid")
            if isinstance(qid, str) and qid:
                done.add(qid)
    return done

_write_lock = threading.Lock()

def append_jsonl(out_path: str, obj: Dict[str, Any]) -> None:
    s = json.dumps(obj, ensure_ascii=False) + "\n"
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(s)

# -----------------------------
# Eval per qid (worker)
# -----------------------------

def eval_one_qid_from_prebuilt(
    qid: str,
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    ndcg_ks: List[int],
) -> Dict[str, Any]:
    if not query:
        query = "(missing query text)"

    true_order = [c["agent_id"] for c in candidates]
    cid_to_agent = {c["candidate_id"]: c["agent_id"] for c in candidates}

    prompt = build_rerank_prompt(query, candidates)
    gpt_raw = _gpt_chat(prompt, model_name=model_name)
    parsed = parse_json_strict(gpt_raw)

    ranking = parsed.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("GPT JSON missing 'ranking' list.")

    # normalize: keep only known cids, keep order, dedupe
    seen_cids = set()
    ranking_clean: List[str] = []
    for x in ranking:
        if not isinstance(x, str):
            continue
        if x not in cid_to_agent:
            continue
        if x in seen_cids:
            continue
        seen_cids.add(x)
        ranking_clean.append(x)

    # if GPT didn't include all candidates, append missing at the end (but we also record this behavior via raw/parsed)
    for c in candidates:
        cid = c["candidate_id"]
        if cid not in seen_cids:
            ranking_clean.append(cid)
            seen_cids.add(cid)

    if len(ranking_clean) != len(candidates):
        raise ValueError("Ranking post-process length mismatch (should not happen).")

    pred_order_agents: List[str] = []
    for cid in ranking_clean:
        if cid in cid_to_agent:
            pred_order_agents.append(cid_to_agent[cid])

    remain = [a for a in true_order if a not in set(pred_order_agents)]
    pred_order_agents.extend(remain)

    top1_match = (pred_order_agents[0] == true_order[0])
    rho = spearman_rho(true_order, pred_order_agents)
    tau = kendall_tau(true_order, pred_order_agents)
    ndcgs = {f"ndcg@{k}": ndcg_at_k(true_order, pred_order_agents, k) for k in ndcg_ks}

    return {
        "qid": qid,
        "query": query,
        "n_candidates": len(candidates),
        "true_order": true_order,
        "gpt_order": pred_order_agents,
        "metrics": {
            "top1_match": top1_match,
            "spearman_rho": rho,
            "kendall_tau": tau,
            **ndcgs,
        },
        "candidates": [{k: v for k, v in c.items() if k != "_true_rank_key"} for c in candidates],
        "gpt": {
            "model": model_name,
            "raw": gpt_raw,
            "parsed": parsed,
        },
    }

def _worker(task: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Return: (qid, result_or_none, error_str_or_none)
    """
    qid = task["qid"]
    try:
        res = eval_one_qid_from_prebuilt(
            qid=qid,
            query=task["query"],
            candidates=task["candidates"],
            model_name=task["model_name"],
            ndcg_ks=task["ndcg_ks"],
        )
        return qid, res, None
    except Exception as e:
        return qid, None, f"{type(e).__name__}: {e}"

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_result_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True, help="jsonl output path (incremental append)")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--ndcg_ks", type=str, default="1,3,5")
    ap.add_argument("--resume", action="store_true", help="skip qids already in out_path")
    ap.add_argument("--max_qids", type=int, default=0, help="0 means no limit; otherwise evaluate at most N eligible qids (>=2 answers)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent GPT requests")
    args = ap.parse_args()

    ndcg_ks: List[int] = []
    for x in args.ndcg_ks.split(","):
        x = x.strip()
        if x:
            try:
                ndcg_ks.append(int(x))
            except Exception:
                pass
    if not ndcg_ks:
        ndcg_ks = [1, 3, 5]

    groups = load_runs_grouped_by_qid(args.run_result_path)
    all_qids = sorted(groups.keys())
    done = load_done_qids(args.out_path) if args.resume else set()

    seen = 0
    evaluated = 0
    skipped_not_enough = 0
    skipped_done = 0
    errors = 0

    # 1) 先预构建 task（>=2 answers 才入队），避免浪费请求
    tasks: List[Dict[str, Any]] = []
    for qid in all_qids:
        seen += 1
        if args.resume and qid in done:
            skipped_done += 1
            continue

        records = groups[qid]
        query = extract_query_text(records[0]) if records else ""
        candidates = build_candidates(records)

        if len(candidates) < 2:
            skipped_not_enough += 1
            continue

        tasks.append({
            "qid": qid,
            "query": query,
            "candidates": candidates,
            "model_name": args.model,
            "ndcg_ks": ndcg_ks,
        })

        if args.max_qids > 0 and len(tasks) >= args.max_qids:
            break

    # 2) 并发执行 GPT 请求（最多 workers=4）
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_to_qid = {ex.submit(_worker, t): t["qid"] for t in tasks}

        for fut in as_completed(future_to_qid):
            qid = future_to_qid[fut]
            qid2, res, err = fut.result()

            if err is None and res is not None:
                append_jsonl(args.out_path, res)
                evaluated += 1
                print(f"[OK] qid={qid2} n={res['n_candidates']} saved -> {args.out_path}")
            else:
                errors += 1
                append_jsonl(args.out_path, {"qid": qid2, "error": err or "Unknown error"})
                print(f"[ERR] qid={qid2}: {err}")

    print("\n" + "=" * 100)
    print(f"Done. seen={seen}")
    print(f"eligible(>=2 answers) submitted={len(tasks)}")
    print(f"evaluated(saved)={evaluated}")
    print(f"skipped_not_enough_answers(<2)={skipped_not_enough}")
    print(f"skipped_done(resume)={skipped_done}")
    print(f"errors={errors}")
    print(f"out={args.out_path}")

if __name__ == "__main__":
    main()
