#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the Section9.1 EasyRec LoRA model on benchmark 2.

This training script is intentionally limited to the delivered LoRA setting:
- benchmark 2 PartI/II/III + Tools
- random negatives only
- LoRA on query/value in the last 4 encoder layers
- fixed 3-epoch fine-tuning
"""

import argparse
import csv
import gzip
import json
import logging
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import save_torch_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from transformers import AutoConfig, AutoTokenizer

from EasyRec_LoRA_model import Easyrec

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    _PEFT_AVAILABLE = False


def relax_legacy_torch_load_guard():
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    modeling_utils.check_torch_load_is_safe = lambda: None
    import_utils.check_torch_load_is_safe = lambda: None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("section9_1_easyrec_lora_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(save_dir, "training.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def metrics_csv_header():
    return ["epoch", "train_loss"]


def write_metrics_row(csv_path: str, row: Dict[str, float]):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_csv_header())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(safe_text(item) for item in value if safe_text(item))
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).strip()
    except Exception:
        return str(value).strip()


def try_lookup_tool_meta(tools_map: Dict[str, Any], token_any: Any):
    raw = safe_text(token_any)
    cand_keys = [raw]
    inner = raw
    if raw.startswith("<<") and raw.endswith(">>"):
        inner = raw[2:-2].strip()
        cand_keys.append(inner)
    if "&&" in inner:
        left, right = inner.split("&&", 1)
        cand_keys.extend([left.strip(), right.strip()])

    for key in cand_keys:
        if not key:
            continue
        meta = tools_map.get(key)
        if meta is None:
            continue
        if isinstance(meta, dict):
            name = safe_text(meta.get("name")) or key
            desc = safe_text(meta.get("description"))
            if not desc:
                for field in ("api_name", "type", "parameters", "expressions"):
                    desc = safe_text(meta.get(field))
                    if desc:
                        break
        else:
            name = key
            desc = safe_text(meta)
        return name, desc
    return None, None


def build_agent_text(agent_obj: Dict[str, Any], tools_map: Dict[str, Any]) -> str:
    model_name = safe_text((agent_obj.get("M", {}) or {}).get("name")) or "Not Provided."
    tool_tokens = list(((agent_obj.get("T", {}) or {}).get("tools", []) or []))
    tool_entries = []
    for token in tool_tokens:
        name, desc = try_lookup_tool_meta(tools_map, token)
        if name:
            tool_entries.append(f"{name} {desc}".strip())
        else:
            tool_entries.append(safe_text(token))
    tool_text = " | ".join(item for item in tool_entries if item)
    text = (model_name + " || " + tool_text).strip(" |")
    return text or model_name


def collect_benchmark_data(data_root: str):
    root = Path(data_root).resolve()
    parts = ["PartI", "PartII", "PartIII"]
    all_agents, all_questions, all_rankings = {}, {}, {}
    for part in parts:
        part_root = root / part
        agents = load_json(part_root / "agents" / "merge.json")
        questions = load_json(part_root / "questions" / "merge.json")
        rankings_obj = load_json(part_root / "rankings" / "merge.json")
        rankings = rankings_obj["rankings"] if isinstance(rankings_obj, dict) and "rankings" in rankings_obj else rankings_obj
        all_agents.update(agents)
        all_questions.update(questions)
        all_rankings.update(rankings)
    tools = load_json(root / "Tools" / "merge.json")
    return all_agents, all_questions, all_rankings, tools


def build_benchmark_corpora(all_agents, all_questions, all_rankings, tools):
    q_ids = [qid for qid in all_questions if qid in all_rankings]
    q_texts = [(all_questions[qid].get("input", "") or "").strip() for qid in q_ids]
    a_ids = list(all_agents.keys())
    a_texts = [build_agent_text(all_agents[aid], tools) for aid in a_ids]
    return q_ids, q_texts, a_ids, a_texts


def save_triples_npy(triples_idx: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            np.save(f, triples_idx.astype(np.int32), allow_pickle=False)
    else:
        np.save(path, triples_idx.astype(np.int32), allow_pickle=False)


def load_triples_npy(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            arr = np.load(f, allow_pickle=False)
    else:
        arr = np.load(path, allow_pickle=False)
    return arr.astype(np.int32)


def build_random_triples_from_rankings(
    *,
    q_ids: Sequence[str],
    all_rankings: Dict[str, List[str]],
    qid_to_idx: Dict[str, int],
    aid_to_idx: Dict[str, int],
    a_ids: Sequence[str],
    train_pos_topk: int,
    rand_neg_per_pos: int,
    seed: int,
):
    rng = random.Random(seed)
    all_aids = list(a_ids)
    a_set = set(all_aids)
    triples_idx: List[Tuple[int, int, int]] = []

    for qid in tqdm(q_ids, desc="Assemble triples", dynamic_ncols=True):
        pos_list = [aid for aid in (all_rankings.get(qid, []) or [])[:train_pos_topk] if aid in a_set]
        if not pos_list:
            continue
        pos_set = set(pos_list)
        rand_pool = [aid for aid in all_aids if aid not in pos_set]
        if not rand_pool:
            continue

        q_idx = qid_to_idx[qid]
        for pos_aid in pos_list:
            sample_k = min(rand_neg_per_pos, len(rand_pool))
            for neg_aid in rng.sample(rand_pool, sample_k):
                triples_idx.append((q_idx, aid_to_idx[pos_aid], aid_to_idx[neg_aid]))

    rng.shuffle(triples_idx)
    return np.asarray(triples_idx, dtype=np.int32)


class TripletIdxDataset(Dataset):
    def __init__(self, triples_idx: np.ndarray, q_texts: Sequence[str], a_texts: Sequence[str]):
        self.arr = triples_idx.astype(np.int32)
        self.q_texts = list(q_texts)
        self.a_texts = list(a_texts)

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        q_idx, p_idx, n_idx = map(int, self.arr[idx])
        return self.q_texts[q_idx], self.a_texts[p_idx], self.a_texts[n_idx]


def collate_triplets(batch):
    q, p, n = [], [], []
    for item in batch:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            continue
        q.append((item[0] or "").strip())
        p.append((item[1] or "").strip())
        n.append((item[2] or "").strip())
    return q, p, n


def make_easyrec_model_args(temp: float, pooler_type: str):
    return SimpleNamespace(
        temp=temp,
        pooler_type=pooler_type,
        do_mlm=False,
        mlm_weight=0.1,
        mlp_only_train=False,
    )


def restrict_lora_to_last_layers(model, last_n_layers: int):
    if last_n_layers <= 0:
        return
    base = model.base_model.model if isinstance(model, PeftModel) else model
    total_layers = len(base.roberta.encoder.layer)
    keep_layers = set(range(max(0, total_layers - last_n_layers), total_layers))
    layer_pat = re.compile(r"\broberta\.encoder\.layer\.(\d+)\.")

    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        match = layer_pat.search(name)
        if match is None or int(match.group(1)) not in keep_layers:
            param.requires_grad = False


def prepare_easyrec_lora_model(model_name: str, temp: float, pooler_type: str):
    if not _PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not installed; cannot run Section9.1 EasyRec LoRA training.")

    relax_legacy_torch_load_guard()
    model_args = make_easyrec_model_args(temp=temp, pooler_type=pooler_type)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.temp = model_args.temp
    cfg.pooler_type = model_args.pooler_type
    cfg.do_mlm = False
    cfg.mlm_weight = 0.1
    cfg.mlp_only_train = False

    model = Easyrec.from_pretrained(
        model_name,
        config=cfg,
        model_args=model_args,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.config.use_cache = False
    model.config.architectures = ["Easyrec"]

    for param in model.lm_head.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_cfg)
    restrict_lora_to_last_layers(model, last_n_layers=4)
    return model, tokenizer


def unwrap_easyrec_model(model):
    if isinstance(model, PeftModel):
        return model.base_model.model
    return model


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def select_amp_dtype(opt: str):
    opt = opt.lower()
    if opt == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
        if torch.cuda.is_available():
            return "fp16"
        return "none"
    return opt


def tokenize_texts(tokenizer, texts: Sequence[str], max_len: int, device: torch.device):
    tokenized = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        return_token_type_ids=False,
    )
    return {k: v.to(device, non_blocking=True) for k, v in tokenized.items() if k in ("input_ids", "attention_mask")}


def create_optimizer(model, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)


def train_one_epoch(
    model,
    tokenizer,
    dataloader: DataLoader,
    optimizer,
    *,
    accum_steps: int,
    max_len: int,
    amp: str,
    max_grad_norm: float,
    device: torch.device,
):
    amp_mode = select_amp_dtype(amp)
    use_fp16 = amp_mode == "fp16"
    use_bf16 = amp_mode == "bf16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    model.train()
    call_model = unwrap_easyrec_model(model)
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    running = 0.0
    steps = 0

    batch_bar = tqdm(dataloader, desc="Train", dynamic_ncols=True, leave=False)
    for q, p, n in batch_bar:
        if not q or not p or not n:
            continue

        q_in = tokenize_texts(tokenizer, q, max_len, device)
        p_in = tokenize_texts(tokenizer, p, max_len, device)
        n_in = tokenize_texts(tokenizer, n, max_len, device)

        if use_bf16:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif use_fp16:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)

        with autocast_ctx:
            outputs = call_model(
                user_input_ids=q_in["input_ids"],
                user_attention_mask=q_in["attention_mask"],
                pos_item_input_ids=p_in["input_ids"],
                pos_item_attention_mask=p_in["attention_mask"],
                neg_item_input_ids=n_in["input_ids"],
                neg_item_attention_mask=n_in["attention_mask"],
                return_dict=True,
            )
            loss = outputs.loss / max(1, accum_steps)

        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        pending += 1
        if pending >= accum_steps:
            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

        running += loss.item() * max(1, accum_steps)
        steps += 1
        batch_bar.set_postfix(
            loss=f"{running / max(1, steps):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    if pending > 0:
        if use_fp16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running / max(1, steps)


def save_full_easyrec_checkpoint(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.config.use_cache = False
    model.config.architectures = ["Easyrec"]
    shared_to_discard = list(getattr(model, "_tied_weights_keys", []) or [])
    save_torch_model(
        model,
        out_dir,
        safe_serialization=True,
        shared_tensors_to_discard=shared_to_discard,
    )
    model.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def export_peft_to_full_model(*, model_name: str, adapter_dir: str, out_dir: str, tokenizer, pooler_type: str, temp: float):
    relax_legacy_torch_load_guard()
    model_args = make_easyrec_model_args(temp=temp, pooler_type=pooler_type)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.temp = model_args.temp
    cfg.pooler_type = model_args.pooler_type
    cfg.do_mlm = False
    cfg.mlm_weight = 0.1
    cfg.mlp_only_train = False
    base_model = Easyrec.from_pretrained(
        model_name,
        config=cfg,
        model_args=model_args,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    merged = merged.merge_and_unload()
    merged.config.use_cache = False
    merged.config.architectures = ["Easyrec"]
    save_full_easyrec_checkpoint(merged, tokenizer, out_dir)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="hkuds/easyrec-roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=640)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", type=str, default="auto", choices=["auto", "bf16", "fp16", "none"])
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--pooler_type", type=str, default="cls", choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"])
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--train_pos_topk", type=int, default=5)
    ap.add_argument("--rand_neg_per_pos", type=int, default=1)
    ap.add_argument("--triples_cache", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)
    save_json(vars(args), os.path.join(args.save_dir, "run_config.json"))
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("[device] %s", device)
    logger.info(
        "[config] model=%s benchmark=%s epochs=%d lr=%.2e batch=%d accum=%d",
        args.model_name,
        args.benchmark_root,
        args.epochs,
        args.lr,
        args.batch_size,
        args.accum_steps,
    )

    all_agents, all_questions, all_rankings, tools = collect_benchmark_data(args.benchmark_root)
    q_ids, q_texts, a_ids, a_texts = build_benchmark_corpora(all_agents, all_questions, all_rankings, tools)
    qid_to_idx = {qid: idx for idx, qid in enumerate(q_ids)}
    aid_to_idx = {aid: idx for idx, aid in enumerate(a_ids)}
    logger.info("[benchmark] agents=%d questions=%d", len(a_ids), len(q_ids))

    if args.triples_cache and os.path.exists(args.triples_cache):
        triples_idx = load_triples_npy(args.triples_cache)
        logger.info("[cache] load triples -> %s (%d)", args.triples_cache, int(triples_idx.shape[0]))
    else:
        logger.info("[prep] assembling training triples from rankings")
        triples_idx = build_random_triples_from_rankings(
            q_ids=q_ids,
            all_rankings=all_rankings,
            qid_to_idx=qid_to_idx,
            aid_to_idx=aid_to_idx,
            a_ids=a_ids,
            train_pos_topk=args.train_pos_topk,
            rand_neg_per_pos=args.rand_neg_per_pos,
            seed=args.seed,
        )
        logger.info("[train] triples=%d", int(triples_idx.shape[0]))
        if args.triples_cache:
            save_triples_npy(triples_idx, args.triples_cache)
            logger.info("[cache] save triples -> %s", args.triples_cache)

    train_ds = TripletIdxDataset(triples_idx, q_texts, a_texts)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_triplets,
    )
    logger.info("[train] num_triples=%d", len(train_ds))

    model, tokenizer = prepare_easyrec_lora_model(
        model_name=args.model_name,
        temp=args.temp,
        pooler_type=args.pooler_type,
    )
    model.to(device)
    optimizer = create_optimizer(model, args.lr)

    trainable, total, pct = count_trainable_parameters(model)
    logger.info("[params] trainable=%d total=%d trainable_pct=%.4f", trainable, total, pct)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    metrics_csv = os.path.join(args.save_dir, "metrics.csv")
    history = []
    for epoch_idx in trange(args.epochs, desc="Epochs", dynamic_ncols=True):
        epoch_num = epoch_idx + 1
        train_loss = train_one_epoch(
            model,
            tokenizer,
            train_dl,
            optimizer,
            accum_steps=args.accum_steps,
            max_len=args.max_len,
            amp=args.amp,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )
        write_metrics_row(metrics_csv, {"epoch": epoch_num, "train_loss": train_loss})
        history.append({"epoch": epoch_num, "train_loss": train_loss})
        save_json(history, os.path.join(args.save_dir, "metrics.json"))
        logger.info("EPOCH %d train_loss=%.6f", epoch_num, train_loss)

    final_adapter_dir = os.path.join(args.save_dir, "final_adapter")
    final_dir = os.path.join(args.save_dir, "final")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    export_peft_to_full_model(
        model_name=args.model_name,
        adapter_dir=final_adapter_dir,
        out_dir=final_dir,
        tokenizer=tokenizer,
        pooler_type=args.pooler_type,
        temp=args.temp,
    )

    summary = {
        "model_name": args.model_name,
        "benchmark_root": args.benchmark_root,
        "use_lora": True,
        "epochs": args.epochs,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": pct,
        "train_qids": len(q_ids),
        "train_triples": int(triples_idx.shape[0]),
        "final_checkpoint": final_dir,
        "final_adapter": final_adapter_dir,
    }
    save_json(summary, os.path.join(args.save_dir, "summary.json"))
    logger.info("[done] summary=%s", os.path.join(args.save_dir, "summary.json"))


if __name__ == "__main__":
    main()
