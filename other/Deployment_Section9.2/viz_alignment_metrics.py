#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Style (NeurIPS-ish)
# -------------------------
def set_neurips_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# -------------------------
# IO
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def dedup_by_qid(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If duplicate qid exists, keep the one with larger n_candidates;
    if tie, keep the later one (stable overwrite).
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        qid = r.get("qid")
        if not isinstance(qid, str) or not qid:
            continue
        n = r.get("n_candidates", None)
        n = int(n) if isinstance(n, int) else -1

        if qid not in best:
            best[qid] = r
        else:
            prev = best[qid]
            prev_n = prev.get("n_candidates", None)
            prev_n = int(prev_n) if isinstance(prev_n, int) else -1
            if n > prev_n:
                best[qid] = r
            elif n == prev_n:
                best[qid] = r  # keep later
    return list(best.values())


# -------------------------
# Ranking metrics (computed from orders)
# -------------------------
METRIC_KEYS = ["top1_match", "spearman_rho", "kendall_tau", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]


def _stable_int_from_str(s: str) -> int:
    # deterministic hash -> int (cross-run stable)
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def extract_order(row: Dict[str, Any], key_candidates: List[str]) -> List[Any]:
    """
    Try keys in order; return list if found, else [].
    """
    for k in key_candidates:
        v = row.get(k, None)
        if isinstance(v, list) and len(v) > 0:
            return v
    return []


def align_orders(expected: List[Any], pred: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Align to intersection of items; keep their original relative orders.
    """
    exp_set = set(expected)
    pred_set = set(pred)
    common = exp_set.intersection(pred_set)
    if not common:
        return [], []
    expected_aligned = [x for x in expected if x in common]
    pred_aligned = [x for x in pred if x in common]
    return expected_aligned, pred_aligned


def spearman_rho_from_ranks(rank_a: Dict[Any, int], rank_b: Dict[Any, int], items: List[Any]) -> float:
    n = len(items)
    if n < 2:
        return np.nan
    # rho = 1 - 6 * sum(d^2) / (n(n^2-1))
    d2 = 0.0
    for it in items:
        d = rank_a[it] - rank_b[it]
        d2 += d * d
    denom = n * (n * n - 1)
    return float(1.0 - (6.0 * d2) / denom) if denom != 0 else np.nan


def kendall_tau_a_from_ranks(rank_a: Dict[Any, int], rank_b: Dict[Any, int], items: List[Any]) -> float:
    """
    Kendall tau-a (no tie handling needed if ranks are permutations).
    """
    n = len(items)
    if n < 2:
        return np.nan
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i = rank_a[items[i]]
            a_j = rank_a[items[j]]
            b_i = rank_b[items[i]]
            b_j = rank_b[items[j]]
            s1 = a_i - a_j
            s2 = b_i - b_j
            if s1 * s2 > 0:
                concordant += 1
            elif s1 * s2 < 0:
                discordant += 1
            else:
                # ties shouldn't happen; ignore defensively
                pass
    total = n * (n - 1) / 2
    return float((concordant - discordant) / total) if total > 0 else np.nan


def ndcg_at_k(expected_order: List[Any], pred_order: List[Any], k: int) -> float:
    """
    Use expected rank as graded relevance:
      rel(item) = 1 / log2(rank_expected + 2)
    Compute nDCG@k for pred ranking.
    """
    if k <= 0:
        return np.nan

    expected_order, pred_order = align_orders(expected_order, pred_order)
    if len(expected_order) == 0:
        return np.nan

    # relevance from expected position
    rel = {}
    for i, it in enumerate(expected_order):
        rel[it] = 1.0 / np.log2(i + 2.0)

    def dcg(order: List[Any]) -> float:
        s = 0.0
        for i, it in enumerate(order[:k]):
            gain = rel.get(it, 0.0)
            s += gain / np.log2(i + 2.0)
        return float(s)

    dcg_val = dcg(pred_order)
    idcg_val = dcg(expected_order)  # ideal is expected itself
    return float(dcg_val / idcg_val) if idcg_val > 0 else np.nan


def compute_metrics_from_orders(expected_order: List[Any], pred_order: List[Any]) -> Dict[str, float]:
    expected_order, pred_order = align_orders(expected_order, pred_order)
    n = len(expected_order)
    if n < 2:
        return {k: np.nan for k in METRIC_KEYS}

    top1 = np.nan
    if expected_order and pred_order:
        top1 = float(1.0 if expected_order[0] == pred_order[0] else 0.0)

    rank_exp = {it: i for i, it in enumerate(expected_order)}
    rank_pred = {it: i for i, it in enumerate(pred_order)}
    items = expected_order[:]  # consistent item list

    out = {
        "top1_match": top1,
        "spearman_rho": spearman_rho_from_ranks(rank_exp, rank_pred, items),
        "kendall_tau": kendall_tau_a_from_ranks(rank_exp, rank_pred, items),
        "ndcg@1": ndcg_at_k(expected_order, pred_order, 1),
        "ndcg@3": ndcg_at_k(expected_order, pred_order, 3),
        "ndcg@5": ndcg_at_k(expected_order, pred_order, 5),
        "ndcg@10": ndcg_at_k(expected_order, pred_order, 10),
    }
    return out


def make_random_expected_order(rec_expected_order: List[Any], qid: str) -> List[Any]:
    rng = np.random.default_rng(_stable_int_from_str(qid))
    arr = list(rec_expected_order)
    rng.shuffle(arr)
    return arr


# -------------------------
# Flatten / DataFrame
# -------------------------
def rows_to_df_from_orders(
    rows: List[Dict[str, Any]],
    expected_key_candidates: List[str],
    pred_key_candidates: List[str],
    random_expected: bool = False,
) -> pd.DataFrame:
    flat = []
    for r in rows:
        if "error" in r and r["error"]:
            continue

        qid = r.get("qid", "")
        if not isinstance(qid, str) or not qid:
            continue

        rec_expected_order = extract_order(r, expected_key_candidates)
        gpt_order = extract_order(r, pred_key_candidates)

        if not rec_expected_order or not gpt_order:
            continue

        expected_order = rec_expected_order
        if random_expected:
            expected_order = make_random_expected_order(rec_expected_order, qid)

        # align + measure effective n_candidates on intersection
        exp_aligned, pred_aligned = align_orders(expected_order, gpt_order)
        n_candidates = len(exp_aligned)
        if n_candidates < 2:
            continue

        metrics = compute_metrics_from_orders(expected_order, gpt_order)

        rec = {
            "qid": qid,
            "n_candidates": int(n_candidates),
        }
        for k in METRIC_KEYS:
            v = metrics.get(k, np.nan)
            if k == "top1_match":
                rec[k] = int(v) if isinstance(v, (int, float)) and not np.isnan(v) else np.nan
            else:
                rec[k] = float(v) if isinstance(v, (int, float)) else np.nan

        flat.append(rec)

    return pd.DataFrame(flat)


# -------------------------
# Stats helpers
# -------------------------
def mean_ci_bootstrap(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=x.size, replace=True)
        means.append(np.mean(samp))
    means = np.array(means)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return (float(np.mean(x)), float(lo), float(hi))


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in METRIC_KEYS:
        m, lo, hi = mean_ci_bootstrap(df[k].to_numpy(dtype=float))
        rows.append({"metric": k, "mean": m, "ci_lo": lo, "ci_hi": hi, "n": int(df[k].notna().sum())})
    return pd.DataFrame(rows)


def summarize_by_n_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for n, g in df.groupby("n_candidates"):
        row = {"n_candidates": int(n), "count_qids": int(len(g))}
        for k in METRIC_KEYS:
            m, lo, hi = mean_ci_bootstrap(g[k].to_numpy(dtype=float))
            row[f"{k}_mean"] = m
            row[f"{k}_ci_lo"] = lo
            row[f"{k}_ci_hi"] = hi
        out.append(row)
    return pd.DataFrame(out).sort_values("n_candidates")


# -------------------------
# Plotting
# -------------------------
def savefig(out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close()


def plot_hist_n_candidates(df: pd.DataFrame, out_dir: Path, suffix: str):
    plt.figure(figsize=(5.2, 3.2))
    plt.hist(df["n_candidates"], bins=np.arange(df["n_candidates"].min(), df["n_candidates"].max() + 2) - 0.5)
    plt.xlabel("Number of valid candidate answers (n_candidates)")
    plt.ylabel("Count (qids)")
    plt.title(f"Distribution of n_candidates ({suffix})")
    savefig(out_dir, f"hist_n_candidates_{suffix}")


def plot_metric_by_n_candidates(summary_by_n: pd.DataFrame, metric: str, out_dir: Path, suffix: str):
    x = summary_by_n["n_candidates"].to_numpy()
    mean = summary_by_n[f"{metric}_mean"].to_numpy(dtype=float)
    lo = summary_by_n[f"{metric}_ci_lo"].to_numpy(dtype=float)
    hi = summary_by_n[f"{metric}_ci_hi"].to_numpy(dtype=float)
    yerr = np.vstack([mean - lo, hi - mean])

    plt.figure(figsize=(5.6, 3.3))
    plt.bar(x, mean)
    plt.errorbar(x, mean, yerr=yerr, fmt="none", capsize=3)
    plt.xlabel("n_candidates")
    plt.ylabel(metric)
    plt.title(f"{metric} vs n_candidates ({suffix}, mean ± 95% CI)")
    savefig(out_dir, f"bar_{metric}_by_n_{suffix}")


def plot_ndcg_curve_overall(df: pd.DataFrame, out_dir: Path, suffix: str):
    ks = [1, 3, 5, 10]
    ys, ylo, yhi = [], [], []
    for k in ks:
        metric = f"ndcg@{k}"
        m, lo, hi = mean_ci_bootstrap(df[metric].to_numpy(dtype=float))
        ys.append(m); ylo.append(lo); yhi.append(hi)

    plt.figure(figsize=(5.0, 3.2))
    plt.plot(ks, ys, marker="o")
    plt.fill_between(ks, ylo, yhi, alpha=0.15)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("nDCG@k")
    plt.title(f"Overall nDCG@k ({suffix}, mean ± 95% CI)")
    savefig(out_dir, f"ndcg_curve_overall_{suffix}")


def plot_scatter_spearman_kendall(df: pd.DataFrame, out_dir: Path, suffix: str):
    plt.figure(figsize=(4.6, 3.6))
    x = df["spearman_rho"].to_numpy(dtype=float)
    y = df["kendall_tau"].to_numpy(dtype=float)
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel("Spearman ρ")
    plt.ylabel("Kendall τ")
    plt.title(f"Correlation metrics scatter ({suffix})")
    savefig(out_dir, f"scatter_spearman_vs_kendall_{suffix}")


def plot_top1_by_n_candidates(df: pd.DataFrame, out_dir: Path, suffix: str):
    g = df.groupby("n_candidates")["top1_match"].agg(["mean", "count"]).reset_index()
    x = g["n_candidates"].to_numpy()
    y = g["mean"].to_numpy(dtype=float)

    plt.figure(figsize=(5.6, 3.3))
    plt.bar(x, y)
    plt.ylim(0, 1.0)
    plt.xlabel("n_candidates")
    plt.ylabel("Top-1 agreement rate")
    plt.title(f"Top-1 alignment vs n_candidates ({suffix})")
    savefig(out_dir, f"bar_top1_rate_by_n_{suffix}")


def plot_overall_metric_compare(overall_rec: pd.DataFrame, overall_rand: pd.DataFrame, out_dir: Path):
    """
    Compare overall means: REC vs GPT  vs  RAND-expected vs GPT
    """
    dfm = overall_rec[["metric", "mean"]].rename(columns={"mean": "rec_mean"}).merge(
        overall_rand[["metric", "mean"]].rename(columns={"mean": "rand_mean"}),
        on="metric",
        how="inner",
    ).sort_values("metric")

    metrics = dfm["metric"].tolist()
    x = np.arange(len(metrics))
    w = 0.38

    plt.figure(figsize=(7.2, 3.6))
    plt.bar(x - w/2, dfm["rec_mean"].to_numpy(dtype=float), width=w, label="REC expected vs GPT")
    plt.bar(x + w/2, dfm["rand_mean"].to_numpy(dtype=float), width=w, label="Random expected vs GPT")
    plt.xticks(x, metrics, rotation=30, ha="right")
    plt.ylabel("Mean metric value")
    plt.title("Overall metrics comparison")
    plt.legend()
    savefig(out_dir, "overall_metric_compare_rec_vs_rand")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl files")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dedup", action="store_true", help="deduplicate by qid (recommended)")
    args = ap.parse_args()

    set_neurips_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    all_rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        all_rows.extend(read_jsonl(p))

    if args.dedup:
        all_rows = dedup_by_qid(all_rows)

    # ---- Key rename (compat):
    #   true_order -> rec_expected_order
    #   gpt_order  -> gpt_evaluated_end2end_performance_order
    expected_keys = ["rec_expected_order", "true_order"]
    pred_keys = ["gpt_evaluated_end2end_performance_order", "gpt_order"]

    # Main comparison: rec_expected_order vs gpt_evaluated_end2end_performance_order
    df_rec = rows_to_df_from_orders(all_rows, expected_keys, pred_keys, random_expected=False)

    # Baseline: random_expected_order (shuffle rec_expected_order) vs gpt_evaluated_end2end_performance_order
    df_rand = rows_to_df_from_orders(all_rows, expected_keys, pred_keys, random_expected=True)

    # Save raw flattened
    df_rec.to_csv(out_dir / "flattened_metrics_rec_expected_vs_gpt.csv", index=False)
    df_rand.to_csv(out_dir / "flattened_metrics_random_expected_vs_gpt.csv", index=False)

    # Summaries
    overall_rec = summarize_overall(df_rec)
    overall_rand = summarize_overall(df_rand)
    by_n_rec = summarize_by_n_candidates(df_rec)
    by_n_rand = summarize_by_n_candidates(df_rand)

    overall_rec.to_csv(out_dir / "summary_overall_rec_expected_vs_gpt.csv", index=False)
    overall_rand.to_csv(out_dir / "summary_overall_random_expected_vs_gpt.csv", index=False)
    by_n_rec.to_csv(out_dir / "summary_by_n_candidates_rec_expected_vs_gpt.csv", index=False)
    by_n_rand.to_csv(out_dir / "summary_by_n_candidates_random_expected_vs_gpt.csv", index=False)

    # Compare table (delta)
    compare = overall_rec[["metric", "mean", "ci_lo", "ci_hi", "n"]].rename(columns={
        "mean": "rec_mean", "ci_lo": "rec_ci_lo", "ci_hi": "rec_ci_hi", "n": "rec_n"
    }).merge(
        overall_rand[["metric", "mean", "ci_lo", "ci_hi", "n"]].rename(columns={
            "mean": "rand_mean", "ci_lo": "rand_ci_lo", "ci_hi": "rand_ci_hi", "n": "rand_n"
        }),
        on="metric",
        how="inner"
    )
    compare["delta_rec_minus_rand"] = compare["rec_mean"] - compare["rand_mean"]
    compare.to_csv(out_dir / "summary_overall_compare_rec_vs_rand.csv", index=False)

    # Print tables
    print("\n=== Overall: REC expected vs GPT (mean ± 95% CI) ===")
    print(overall_rec.to_string(index=False))

    print("\n=== Overall: RANDOM expected vs GPT (mean ± 95% CI) ===")
    print(overall_rand.to_string(index=False))

    print("\n=== Overall Compare (REC - RAND) ===")
    print(compare[["metric", "rec_mean", "rand_mean", "delta_rec_minus_rand"]].to_string(index=False))

    # Plots (REC)
    plot_hist_n_candidates(df_rec, out_dir, "rec_expected_vs_gpt")
    plot_top1_by_n_candidates(df_rec, out_dir, "rec_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rec, "spearman_rho", out_dir, "rec_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rec, "kendall_tau", out_dir, "rec_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rec, "ndcg@1", out_dir, "rec_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rec, "ndcg@10", out_dir, "rec_expected_vs_gpt")
    plot_ndcg_curve_overall(df_rec, out_dir, "rec_expected_vs_gpt")
    plot_scatter_spearman_kendall(df_rec, out_dir, "rec_expected_vs_gpt")

    # Plots (RAND baseline)
    plot_hist_n_candidates(df_rand, out_dir, "random_expected_vs_gpt")
    plot_top1_by_n_candidates(df_rand, out_dir, "random_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rand, "spearman_rho", out_dir, "random_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rand, "kendall_tau", out_dir, "random_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rand, "ndcg@1", out_dir, "random_expected_vs_gpt")
    plot_metric_by_n_candidates(by_n_rand, "ndcg@10", out_dir, "random_expected_vs_gpt")
    plot_ndcg_curve_overall(df_rand, out_dir, "random_expected_vs_gpt")
    plot_scatter_spearman_kendall(df_rand, out_dir, "random_expected_vs_gpt")

    # Comparison plot
    plot_overall_metric_compare(overall_rec, overall_rand, out_dir)

    print("\nSaved outputs to:", str(out_dir.resolve()))
    print("Key files:")
    print(" - flattened_metrics_rec_expected_vs_gpt.csv")
    print(" - flattened_metrics_random_expected_vs_gpt.csv")
    print(" - summary_overall_rec_expected_vs_gpt.csv")
    print(" - summary_overall_random_expected_vs_gpt.csv")
    print(" - summary_overall_compare_rec_vs_rand.csv")
    print(" - figures: *.pdf / *.png")


if __name__ == "__main__":
    main()
