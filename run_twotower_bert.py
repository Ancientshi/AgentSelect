#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import stratified_train_valid_split
from agent_rec.features import (
    build_agent_content_view,
    build_agent_tool_id_buffers,
    build_unified_corpora,
    UNK_TOOL_TOKEN,
    UNK_LLM_TOKEN,
)
from agent_rec.models.two_tower import TwoTowerTFIDF as TwoTowerBERT
from agent_rec.run_common import (
    bootstrap_run,
    cache_key_from_meta,
    cache_key_from_text,
    build_pos_pairs,
    load_or_build_training_cache,
    shared_cache_dir,
)

from eval import build_aid_to_llm, evaluate_parts, print_sample_recommendations


def info_nce_loss(qe: torch.Tensor, ae: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = qe @ ae.t()
    labels = torch.arange(qe.size(0), device=qe.device)
    return torch.nn.functional.cross_entropy(logits / temperature, labels)


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear.

    This keeps the original Linear weight frozen and trains only A/B.
    It is intentionally local to this script so no PEFT dependency is needed.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)

        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        dev = base_linear.weight.device
        dt = base_linear.weight.dtype
        self.A = nn.Parameter(torch.empty((self.r, self.in_features), device=dev, dtype=dt))
        self.B = nn.Parameter(torch.empty((self.out_features, self.r), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        h = F.linear(self.dropout(x), self.A)
        lora = F.linear(h, self.B)
        return base + lora * self.scaling


def apply_lora_to_encoder(encoder: nn.Module, target_keywords: List[str], r: int, alpha: int, dropout: float) -> None:
    replaced = 0
    for _, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in child_name.lower() for k in target_keywords):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    print(f"[LoRA] injected into {replaced} Linear layers; targets={target_keywords}")


def _get_transformer_layers(model: nn.Module):
    # BERT / DistilBERT / RoBERTa common layouts.
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "bert") and hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "layer"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder") and hasattr(model.roberta.encoder, "layer"):
        return model.roberta.encoder.layer
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        tr = model.distilbert.transformer
        if hasattr(tr, "layer"):
            return tr.layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return model.transformer.layer
    return None


def set_finetune_scope(encoder: nn.Module, unfreeze_last_n: int, unfreeze_emb: bool) -> None:
    for p in encoder.parameters():
        p.requires_grad = False

    layers = _get_transformer_layers(encoder)
    if layers is None:
        print("[warn] could not locate transformer layers; keeping encoder frozen")
        return

    n_layers = len(layers)
    if unfreeze_last_n <= 0 or unfreeze_last_n >= n_layers:
        for p in encoder.parameters():
            p.requires_grad = True
    else:
        for block in layers[-int(unfreeze_last_n):]:
            for p in block.parameters():
                p.requires_grad = True

    if unfreeze_emb:
        for attr in ["embeddings"]:
            if hasattr(encoder, attr):
                for p in getattr(encoder, attr).parameters():
                    p.requires_grad = True
        for root_name in ["bert", "roberta", "distilbert"]:
            root = getattr(encoder, root_name, None)
            if root is not None and hasattr(root, "embeddings"):
                for p in root.embeddings.parameters():
                    p.requires_grad = True
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    encoder,
    device: torch.device,
    max_len: int = 128,
    batch_size: int = 256,
    pooling: str = "cls",
    desc: str = "Encoding BERT",
) -> np.ndarray:
    if not texts:
        dim = getattr(getattr(encoder, "config", None), "hidden_size", 0) or 0
        return np.zeros((0, dim), dtype=np.float32)

    was_training = encoder.training
    encoder.eval()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, dynamic_ncols=True):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = encoder(**toks)
        if hasattr(out, "last_hidden_state"):
            if pooling == "cls":
                vec = out.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                attn = toks["attention_mask"].unsqueeze(-1)
                vec = (out.last_hidden_state * attn).sum(1) / attn.sum(1).clamp(min=1)
            else:
                raise ValueError(f"Unsupported pooling={pooling}")
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            vec = out.pooler_output
        else:
            raise RuntimeError("Transformer output has neither last_hidden_state nor pooler_output")
        embs.append(vec.detach().float().cpu())
    if was_training:
        encoder.train()
    return torch.cat(embs, dim=0).numpy().astype(np.float32)


def encode_batch(
    tokenizer,
    encoder,
    texts: List[str],
    device: torch.device,
    max_len: int = 128,
    pooling: str = "cls",
) -> torch.Tensor:
    if not texts:
        dim = getattr(getattr(encoder, "config", None), "hidden_size", 0) or 0
        return torch.zeros((0, dim), device=device)
    toks = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    out = encoder(**toks)
    if hasattr(out, "last_hidden_state"):
        if pooling == "cls":
            return out.last_hidden_state[:, 0, :]
        if pooling == "mean":
            attn = toks["attention_mask"].unsqueeze(-1)
            return (out.last_hidden_state * attn).sum(1) / attn.sum(1).clamp(min=1)
        raise ValueError(f"Unsupported pooling={pooling}")
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    raise RuntimeError("Transformer output has neither last_hidden_state nor pooler_output")


def ensure_transformer_cache_dir(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def transformer_cache_exists(cache_dir: str) -> bool:
    needed = [
        "q_ids.json",
        "a_ids.json",
        "tool_names.json",
        "tool_id_vocab.json",
        "llm_ids.json",
        "llm_vocab.json",
        "Q_emb.npy",
        "A_model_content.npy",
        "A_tool_content.npy",
        "agent_tool_idx_padded.npy",
        "agent_tool_mask.npy",
        "agent_llm_idx.npy",
        "enc_meta.json",
    ]
    return all(os.path.exists(os.path.join(cache_dir, x)) for x in needed)


def save_transformer_cache(
    cache_dir: str,
    q_ids: List[str],
    a_ids: List[str],
    tool_names: List[str],
    tool_id_vocab: List[str],
    llm_ids: List[str],
    llm_vocab: List[str],
    Q_emb: np.ndarray,
    A_model_emb: np.ndarray,
    A_tool_emb: np.ndarray,
    agent_tool_idx_padded: np.ndarray,
    agent_tool_mask: np.ndarray,
    agent_llm_idx: np.ndarray,
    enc_meta: dict,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for name, obj in [
        ("q_ids.json", q_ids),
        ("a_ids.json", a_ids),
        ("tool_names.json", tool_names),
        ("tool_id_vocab.json", tool_id_vocab),
        ("llm_ids.json", llm_ids),
        ("llm_vocab.json", llm_vocab),
        ("enc_meta.json", enc_meta),
    ]:
        with open(os.path.join(cache_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2 if name == "enc_meta.json" else None)
    np.save(os.path.join(cache_dir, "Q_emb.npy"), Q_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_model_content.npy"), A_model_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_tool_content.npy"), A_tool_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir, "agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_llm_idx.npy"), agent_llm_idx.astype(np.int64))


def load_transformer_cache(cache_dir: str):
    def load_json(name: str):
        with open(os.path.join(cache_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "q_ids": load_json("q_ids.json"),
        "a_ids": load_json("a_ids.json"),
        "tool_names": load_json("tool_names.json"),
        "tool_id_vocab": load_json("tool_id_vocab.json"),
        "llm_ids": load_json("llm_ids.json"),
        "llm_vocab": load_json("llm_vocab.json"),
        "Q_emb": np.load(os.path.join(cache_dir, "Q_emb.npy")),
        "A_model_emb": np.load(os.path.join(cache_dir, "A_model_content.npy")),
        "A_tool_emb": np.load(os.path.join(cache_dir, "A_tool_content.npy")),
        "agent_tool_idx_padded": np.load(os.path.join(cache_dir, "agent_tool_idx_padded.npy")),
        "agent_tool_mask": np.load(os.path.join(cache_dir, "agent_tool_mask.npy")),
        "agent_llm_idx": np.load(os.path.join(cache_dir, "agent_llm_idx.npy")),
        "enc_meta": load_json("enc_meta.json"),
    }


def build_tool_content_emb_for_agents(
    a_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_emb: np.ndarray,
) -> np.ndarray:
    tool_name_to_idx = {t: i for i, t in enumerate(tool_names)}
    if tool_emb.shape[0] == 0:
        return np.zeros((len(a_tool_lists), 0), dtype=np.float32)
    out = []
    for tools_for_agent in a_tool_lists:
        idxs = [tool_name_to_idx[t] for t in tools_for_agent if t in tool_name_to_idx]
        if idxs:
            out.append(tool_emb[idxs].mean(axis=0))
        else:
            out.append(np.zeros((tool_emb.shape[1],), dtype=np.float32))
    return np.stack(out, axis=0).astype(np.float32)


def normalize_np(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def make_a_cpu(
    A_model_emb: np.ndarray,
    A_tool_emb: np.ndarray,
    use_model_content_vector: bool,
    use_tool_content_vector: bool,
) -> np.ndarray:
    # build_agent_content_view supports explicit arrays in the current codebase;
    # if older versions only support cache=..., this fallback keeps the script runnable.
    try:
        return build_agent_content_view(
            A_model_content=A_model_emb,
            A_tool_content=A_tool_emb,
            use_model_content_vector=use_model_content_vector,
            use_tool_content_vector=use_tool_content_vector,
        ).astype(np.float32)
    except TypeError:
        parts = []
        if use_model_content_vector:
            parts.append(A_model_emb)
        if use_tool_content_vector:
            parts.append(A_tool_emb)
        if not parts:
            raise ValueError("Enable at least one of use_model_content_vector/use_tool_content_vector.")
        return np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 else parts[0].astype(np.float32)


def build_frozen_bert_features(
    *,
    cache_dir: str,
    rebuild: bool,
    q_ids: List[str],
    q_texts: List[str],
    a_ids: List[str],
    a_model_names: List[str],
    a_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_texts: List[str],
    tool_id_vocab: List[str],
    llm_ids: List[str],
    llm_vocab: List[str],
    agent_tool_idx_padded: np.ndarray,
    agent_tool_mask: np.ndarray,
    agent_llm_idx: np.ndarray,
    tokenizer,
    encoder,
    device: torch.device,
    pretrained_model: str,
    max_len: int,
    pooling: str,
    encode_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    want_meta = {
        "pretrained_model": pretrained_model,
        "max_len": int(max_len),
        "pooling": pooling,
    }

    if transformer_cache_exists(cache_dir) and not rebuild:
        cache = load_transformer_cache(cache_dir)
        ok = (
            cache["q_ids"] == q_ids
            and cache["a_ids"] == a_ids
            and cache["tool_names"] == tool_names
            and cache["tool_id_vocab"] == tool_id_vocab
            and cache["llm_ids"] == llm_ids
            and cache["llm_vocab"] == llm_vocab
            and cache["enc_meta"].get("pretrained_model") == pretrained_model
            and int(cache["enc_meta"].get("max_len", -1)) == int(max_len)
            and cache["enc_meta"].get("pooling") == pooling
        )
        if ok:
            print(f"[cache] loaded frozen BERT features from {cache_dir}")
            return (
                cache["Q_emb"].astype(np.float32),
                cache["A_model_emb"].astype(np.float32),
                cache["A_tool_emb"].astype(np.float32),
                cache["agent_llm_idx"].astype(np.int64),
            )
        print("[cache] frozen BERT cache mismatch; rebuilding")

    encoder.eval()
    Q_emb = encode_texts(q_texts, tokenizer, encoder, device, max_len, encode_batch_size, pooling, "Encoding queries")
    A_model_emb = encode_texts(
        a_model_names, tokenizer, encoder, device, max_len, encode_batch_size, pooling, "Encoding agent LLM text"
    )
    tool_emb = encode_texts(tool_texts, tokenizer, encoder, device, max_len, encode_batch_size, pooling, "Encoding tools")

    Q_emb = normalize_np(Q_emb)
    A_model_emb = normalize_np(A_model_emb)
    tool_emb = normalize_np(tool_emb)
    A_tool_emb = normalize_np(build_tool_content_emb_for_agents(a_tool_lists, tool_names, tool_emb))

    save_transformer_cache(
        cache_dir,
        q_ids,
        a_ids,
        tool_names,
        tool_id_vocab,
        llm_ids,
        llm_vocab,
        Q_emb,
        A_model_emb,
        A_tool_emb,
        agent_tool_idx_padded,
        agent_tool_mask,
        agent_llm_idx,
        want_meta,
    )
    print(f"[cache] saved frozen BERT features to {cache_dir}")
    return Q_emb, A_model_emb, A_tool_emb, agent_llm_idx


@torch.no_grad()
def build_eval_bert_features(
    *,
    q_texts: List[str],
    a_model_names: List[str],
    a_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_texts: List[str],
    tokenizer,
    encoder,
    device: torch.device,
    max_len: int,
    pooling: str,
    encode_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    Q_emb = encode_texts(q_texts, tokenizer, encoder, device, max_len, encode_batch_size, pooling, "Eval encode queries")
    A_model_emb = encode_texts(
        a_model_names, tokenizer, encoder, device, max_len, encode_batch_size, pooling, "Eval encode agent LLM text"
    )
    tool_emb = encode_texts(tool_texts, tokenizer, encoder, device, max_len, encode_batch_size, pooling, "Eval encode tools")
    Q_emb = normalize_np(Q_emb)
    A_model_emb = normalize_np(A_model_emb)
    tool_emb = normalize_np(tool_emb)
    A_tool_emb = normalize_np(build_tool_content_emb_for_agents(a_tool_lists, tool_names, tool_emb))
    return Q_emb, A_model_emb, A_tool_emb


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="two_tower_bert",
        device_default="cuda:0",
        epochs_default=3,
        batch_size_default=256,
        lr_default=3e-4,
        include_neg_per_pos=False,
    )
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)
    parser.add_argument("--eval_chunk", type=int, default=8192, help="batch size over agents for inference")
    parser.add_argument("--amp", type=int, default=0, help="1 to enable autocast on CUDA (bfloat16)")

    parser.add_argument("--tune_mode", type=str, choices=["frozen", "full", "lora"], default="frozen")
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--encoder_weight_decay", type=float, default=0.0)
    parser.add_argument("--unfreeze_last_n", type=int, default=0, help="for --tune_mode full; 0 means all layers")
    parser.add_argument("--unfreeze_emb", type=int, default=0)
    parser.add_argument("--grad_ckpt", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="query,key,value,dense,q_lin,k_lin,v_lin,out_lin",
        help="comma-separated Linear child-name keywords",
    )

    parser.add_argument("--use_tool_id_emb", type=int, default=1)
    parser.add_argument("--use_llm_id_emb", type=int, default=0, help="1 to add LLM-ID embedding into agent tower")
    parser.add_argument(
        "--use_model_content_vector", type=int, default=1, help="1 to include BERT(model/backbone text) in agent content view"
    )
    parser.add_argument(
        "--use_tool_content_vector", type=int, default=1, help="1 to include BERT(tool descriptions) in agent content view"
    )
    parser.add_argument("--use_tool_emb", type=int, default=None, help="Deprecated alias for --use_tool_id_emb")
    parser.add_argument("--use_agent_id_emb", type=int, default=0, help="1 to add learnable per-agent ID embedding into agent tower")
    parser.add_argument("--use_query_id_emb", type=int, default=0, help="1 to add query-ID embedding into query tower")
    parser.add_argument(
        "--soft_eval", "--soft_eva",
        dest="soft_eval",
        type=int,
        default=0,
        help="1 to evaluate by backbone-LLM match instead of exact agent id (default: 0)",
    )
    args = parser.parse_args()

    use_tool_id_emb = bool(args.use_tool_id_emb if args.use_tool_emb is None else args.use_tool_emb)
    use_llm_id_emb = bool(args.use_llm_id_emb)
    use_agent_id_emb = bool(args.use_agent_id_emb)
    use_model_content_vector = bool(args.use_model_content_vector)
    use_tool_content_vector = bool(args.use_tool_content_vector)
    if not (use_model_content_vector or use_tool_content_vector):
        raise ValueError("Enable at least one of --use_model_content_vector / --use_tool_content_vector.")

    boot = bootstrap_run(data_root=args.data_root, exp_name=args.exp_name, topk=args.topk, with_tools=True)
    bundle = boot.bundle
    tools = boot.tools
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    q_ids = boot.q_ids
    a_ids = boot.a_ids
    qid2idx = boot.qid2idx
    aid2idx = boot.aid2idx
    qids_in_rank = boot.qids_in_rank
    data_sig = boot.data_sig
    exp_cache_dir = boot.exp_cache_dir

    (
        q_ids_u,
        q_texts,
        tool_names,
        tool_texts,
        a_ids_u,
        a_model_names,
        a_tool_lists,
        llm_ids,
    ) = build_unified_corpora(all_agents, all_questions, tools)
    if list(q_ids_u) != list(q_ids) or list(a_ids_u) != list(a_ids):
        raise RuntimeError("ID ordering mismatch between bootstrap_run and build_unified_corpora.")

    tool_id_vocab = [UNK_TOOL_TOKEN] + list(tool_names)
    tool_vocab_map = {n: i for i, n in enumerate(tool_id_vocab)}
    agent_tool_idx_padded_np, agent_tool_mask_np = build_agent_tool_id_buffers(a_tool_lists, tool_vocab_map)

    llm_vocab = [UNK_LLM_TOKEN] + [lid for lid in llm_ids if lid]
    llm_vocab = list(dict.fromkeys(llm_vocab))
    llm_vocab_map = {n: i for i, n in enumerate(llm_vocab)}
    agent_llm_idx_np = np.array([llm_vocab_map.get(lid, 0) for lid in llm_ids], dtype=np.int64)

    aid_to_llm = build_aid_to_llm(
        a_ids=a_ids,
        llm_vocab=list(llm_vocab),
        agent_llm_idx=np.asarray(agent_llm_idx_np, dtype=np.int64),
    )

    train_parts = list(args.train_parts)
    eval_parts = list(args.eval_parts)
    qids_for_training = [qid for qid in qids_in_rank if qid_to_part.get(qid, "") in set(train_parts)]
    if not qids_for_training:
        raise RuntimeError(
            f"No qids found for train_parts={train_parts}. Available parts: {sorted(set(qid_to_part.values()))}"
        )
    print(f"[parts] train_parts={train_parts} -> qids={len(qids_for_training)}; eval_parts={eval_parts}")

    want_meta = {
        "data_sig": data_sig,
        "pos_topk_by_part": POS_TOPK_BY_PART,
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "pair_type": "q_pos_only_posTopK",
        "train_parts": train_parts,
    }
    training_cache_dir = shared_cache_dir(args.data_root, "training", f"{data_sig}_{cache_key_from_meta(want_meta)}")

    def build_cache():
        train_qids, valid_qids = stratified_train_valid_split(
            qids_for_training,
            qid_to_part=qid_to_part,
            valid_ratio=args.valid_ratio,
            seed=args.split_seed,
        )
        print(f"[split] train_parts={train_parts} train={len(train_qids)}  valid={len(valid_qids)}")
        pairs = build_pos_pairs(
            {qid: all_rankings[qid] for qid in train_qids},
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            rng_seed=args.rng_seed_pairs,
        )
        pairs_idx = [(qid2idx[q], aid2idx[a]) for (q, a) in pairs]
        return train_qids, valid_qids, np.array(pairs_idx, dtype=np.int64)

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    bert_encoder = AutoModel.from_pretrained(args.pretrained_model).to(device)

    if args.grad_ckpt:
        try:
            bert_encoder.gradient_checkpointing_enable()
            print("[encoder] gradient checkpointing enabled")
        except Exception as exc:
            print(f"[encoder] gradient checkpointing not supported: {exc}")

    if args.tune_mode == "frozen":
        for p in bert_encoder.parameters():
            p.requires_grad = False
    elif args.tune_mode == "full":
        set_finetune_scope(bert_encoder, args.unfreeze_last_n, bool(args.unfreeze_emb))
    elif args.tune_mode == "lora":
        for p in bert_encoder.parameters():
            p.requires_grad = False
        targets = [x.strip().lower() for x in args.lora_targets.split(",") if x.strip()]
        apply_lora_to_encoder(bert_encoder, targets, args.lora_r, args.lora_alpha, args.lora_dropout)
    else:
        raise ValueError(f"Unknown tune_mode={args.tune_mode}")

    trainable_encoder_params = [p for p in bert_encoder.parameters() if p.requires_grad]
    print(f"[encoder] tune_mode={args.tune_mode}; trainable params={sum(p.numel() for p in trainable_encoder_params):,}")

    bert_key_meta = {
        "data_sig": data_sig,
        "pretrained_model": args.pretrained_model,
        "max_len": int(args.max_len),
        "pooling": args.pooling,
    }
    bert_cache_dir = ensure_transformer_cache_dir(
        shared_cache_dir(args.data_root, "features", f"bert_{cache_key_from_meta(bert_key_meta)}")
    )

    use_embedding_cache = args.tune_mode == "frozen"
    if use_embedding_cache:
        Q_cpu, A_model_emb, A_tool_emb, _ = build_frozen_bert_features(
            cache_dir=bert_cache_dir,
            rebuild=bool(args.rebuild_feature_cache),
            q_ids=q_ids,
            q_texts=q_texts,
            a_ids=a_ids,
            a_model_names=a_model_names,
            a_tool_lists=a_tool_lists,
            tool_names=tool_names,
            tool_texts=tool_texts,
            tool_id_vocab=tool_id_vocab,
            llm_ids=llm_ids,
            llm_vocab=llm_vocab,
            agent_tool_idx_padded=agent_tool_idx_padded_np,
            agent_tool_mask=agent_tool_mask_np,
            agent_llm_idx=agent_llm_idx_np,
            tokenizer=tokenizer,
            encoder=bert_encoder,
            device=device,
            pretrained_model=args.pretrained_model,
            max_len=args.max_len,
            pooling=args.pooling,
            encode_batch_size=args.encode_batch_size,
        )
        A_cpu = make_a_cpu(A_model_emb, A_tool_emb, use_model_content_vector, use_tool_content_vector)
        d_q = int(Q_cpu.shape[1])
        d_a = int(A_cpu.shape[1])
    else:
        bert_encoder.eval()
        with torch.no_grad():
            tmp = encode_batch(tokenizer, bert_encoder, ["dimension probe"], device, args.max_len, args.pooling)
        bert_encoder.train()
        d_text = int(tmp.shape[1])
        d_q = d_text
        d_a = d_text * int(use_model_content_vector) + d_text * int(use_tool_content_vector)
        Q_cpu = None
        A_cpu = None

    encoder = TwoTowerBERT(
        d_q=d_q,
        d_a=d_a,
        num_tools=int(len(tool_id_vocab)),
        num_llm_ids=int(len(llm_vocab)),
        agent_tool_idx_padded=torch.tensor(agent_tool_idx_padded_np, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(agent_tool_mask_np, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(agent_llm_idx_np, dtype=torch.long, device=device),
        hid=args.hid,
        use_tool_id_emb=use_tool_id_emb,
        use_llm_id_emb=use_llm_id_emb,
        num_agents=len(a_ids),
        num_queries=len(q_ids),
        use_query_id_emb=bool(args.use_query_id_emb),
        use_agent_id_emb=use_agent_id_emb,
    ).to(device)

    param_groups = [{"params": list(encoder.parameters()), "lr": args.lr}]
    if trainable_encoder_params:
        param_groups.append(
            {"params": trainable_encoder_params, "lr": args.encoder_lr, "weight_decay": args.encoder_weight_decay}
        )
    optimizer = torch.optim.Adam(param_groups)

    num_pairs = int(pairs_idx_np.shape[0])
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    use_amp = args.amp == 1 and device.type == "cuda"
    if use_embedding_cache:
        Q_t = torch.tensor(Q_cpu, dtype=torch.float32, device=device)
        A_t = torch.tensor(A_cpu, dtype=torch.float32, device=device)
    else:
        Q_t = A_t = None

    tool_name_to_text: Dict[str, str] = {n: t for n, t in zip(tool_names, tool_texts)}

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(pairs_idx_np)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        encoder.train()
        if not use_embedding_cache:
            bert_encoder.train()
        for b in pbar:
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, num_pairs))
            batch = pairs_idx_np[sl]
            if batch.size == 0:
                continue
            q_idx = batch[:, 0]
            a_idx = batch[:, 1]

            q_idx_t = torch.from_numpy(q_idx).long().to(device, non_blocking=True)
            a_idx_t = torch.from_numpy(a_idx).long().to(device, non_blocking=True)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with autocast_ctx:
                if use_embedding_cache:
                    q_vec = Q_t[q_idx_t]
                    a_pos = A_t[a_idx_t]
                else:
                    uniq_q, inv_q = torch.unique(q_idx_t, sorted=True, return_inverse=True)
                    uniq_a, inv_a = torch.unique(a_idx_t, sorted=True, return_inverse=True)

                    q_text_batch = [q_texts[i] for i in uniq_q.tolist()]
                    q_vec_uniq = encode_batch(tokenizer, bert_encoder, q_text_batch, device, args.max_len, args.pooling)
                    q_vec = F.normalize(q_vec_uniq, dim=-1)[inv_q]

                    agent_model_texts = [a_model_names[i] for i in uniq_a.tolist()]
                    model_emb = encode_batch(tokenizer, bert_encoder, agent_model_texts, device, args.max_len, args.pooling)
                    model_emb = F.normalize(model_emb, dim=-1)

                    parts = []
                    if use_model_content_vector:
                        parts.append(model_emb)
                    if use_tool_content_vector:
                        needed_tools = sorted({t for ai in uniq_a.tolist() for t in a_tool_lists[ai] if t in tool_name_to_text})
                        tool_emb_map: Dict[str, torch.Tensor] = {}
                        if needed_tools:
                            tool_text_batch = [tool_name_to_text[t] for t in needed_tools]
                            tool_emb_batch = encode_batch(tokenizer, bert_encoder, tool_text_batch, device, args.max_len, args.pooling)
                            tool_emb_batch = F.normalize(tool_emb_batch, dim=-1)
                            for name, emb in zip(needed_tools, tool_emb_batch):
                                tool_emb_map[name] = emb
                        zero_tool = torch.zeros((model_emb.shape[1],), device=device, dtype=model_emb.dtype)
                        tool_feats = []
                        for ai in uniq_a.tolist():
                            names = [t for t in a_tool_lists[ai] if t in tool_emb_map]
                            if names:
                                tool_feats.append(torch.stack([tool_emb_map[n] for n in names], dim=0).mean(dim=0))
                            else:
                                tool_feats.append(zero_tool)
                        tool_feats_t = torch.stack(tool_feats, dim=0)
                        tool_feats_t = F.normalize(tool_feats_t, dim=-1)
                        parts.append(tool_feats_t)
                    a_vec_uniq = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
                    a_pos = a_vec_uniq[inv_a]

                qe = encoder.encode_q(q_vec, q_idx=q_idx_t)
                ae = encoder.encode_a(a_pos, a_idx_t)
                loss = info_nce_loss(qe, ae, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - InfoNCE: {(total / max(1, num_batches)):.4f}")

    if not use_embedding_cache:
        Q_cpu, A_model_emb, A_tool_emb = build_eval_bert_features(
            q_texts=q_texts,
            a_model_names=a_model_names,
            a_tool_lists=a_tool_lists,
            tool_names=tool_names,
            tool_texts=tool_texts,
            tokenizer=tokenizer,
            encoder=bert_encoder,
            device=device,
            max_len=args.max_len,
            pooling=args.pooling,
            encode_batch_size=args.encode_batch_size,
        )
        A_cpu = make_a_cpu(A_model_emb, A_tool_emb, use_model_content_vector, use_tool_content_vector)

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    save_sig = want_meta["data_sig"]
    model_path = os.path.join(model_dir, f"{args.exp_name}_{save_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{save_sig}.json")

    ckpt = {
        "state_dict": encoder.state_dict(),
        "bert_state_dict": bert_encoder.state_dict() if args.tune_mode in {"full", "lora"} else None,
        "data_sig": save_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q": int(Q_cpu.shape[1]),
            "d_a": int(A_cpu.shape[1]),
            "hid": int(args.hid),
            "num_tools": int(len(tool_id_vocab)),
            "num_llm_ids": int(len(llm_vocab)),
            "num_agents": int(len(a_ids)),
        },
        "parts": {"train_parts": train_parts, "eval_parts": eval_parts},
        "flags": {
            "use_tool_id_emb": use_tool_id_emb,
            "use_llm_id_emb": use_llm_id_emb,
            "use_model_content_vector": use_model_content_vector,
            "use_tool_content_vector": use_tool_content_vector,
            "use_query_id_emb": bool(args.use_query_id_emb),
            "use_agent_id_emb": use_agent_id_emb,
            "tune_mode": args.tune_mode,
        },
        "mappings": {"q_ids": q_ids, "a_ids": a_ids, "tool_names": tool_names, "llm_vocab": llm_vocab},
    }
    torch.save(ckpt, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_sig": save_sig,
                "q_ids": q_ids,
                "a_ids": a_ids,
                "tool_names": tool_names,
                "train_parts": train_parts,
                "eval_parts": eval_parts,
                "pretrained_model": args.pretrained_model,
                "tune_mode": args.tune_mode,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    evaluate_parts(
        encoder=encoder,
        Q_cpu=Q_cpu,
        A_cpu=A_cpu,
        qid2idx=qid2idx,
        a_ids=a_ids,
        all_rankings=all_rankings,
        qids_in_rank=qids_in_rank,
        qid_to_part=qid_to_part,
        eval_parts=eval_parts,
        device=device,
        topk=int(args.topk),
        cand_size=args.eval_cand_size,
        rng_seed=args.rng_seed_pairs,
        amp=use_amp,
        exp_name=args.exp_name,
        soft_eval=bool(args.soft_eval),
        aid_to_llm=aid_to_llm,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
    )

    print_sample_recommendations(
        encoder=encoder,
        Q_cpu=Q_cpu,
        A_cpu=A_cpu,
        qid2idx=qid2idx,
        a_ids=a_ids,
        q_ids=q_ids,
        all_questions=all_questions,
        device=device,
        topk=args.topk,
        chunk=args.eval_chunk,
        num_samples=5,
    )


if __name__ == "__main__":
    main()
