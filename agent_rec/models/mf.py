#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, Optional, Sequence, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from .base import RecommenderBase


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


class MF(RecommenderBase):
    """Matrix-factorization baseline with component-based agent representations.

    Agent/configuration vectors are built from LLM IDs and mean-pooled tool IDs:

        a_i = alpha_llm * EmbLLM(llm_i) + alpha_tool * mean_{t in T_i} EmbTool(t)

    There is intentionally no independent ``nn.Embedding(num_agents, factors)`` for
    agents. By default, no per-agent bias is used either, so the model cannot
    memorize an agent/configuration purely by its agent ID.
    """

    def __init__(
        self,
        num_q: int,
        num_a: int,
        num_llm_ids: int,
        agent_llm_idx: torch.LongTensor,
        factors: int = 128,
        add_bias: bool = True,
        use_llm_id_emb: bool = True,
        *,
        num_tool_ids: int = 0,
        agent_tool_indices_padded: Optional[torch.LongTensor] = None,
        agent_tool_mask: Optional[torch.FloatTensor] = None,
        use_tool_id_emb: bool = True,
        use_agent_bias: bool = False,
        use_agent_id_emb: bool = False,
        alpha_llm: float = 1.0,
        alpha_tool: float = 1.0,
        alpha_agent: float = 1.0,
    ):
        super().__init__()
        self.num_a = int(num_a)
        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        self.use_tool_id_emb = bool(use_tool_id_emb) and num_tool_ids > 0
        self.use_agent_bias = bool(use_agent_bias)
        self.use_agent_id_emb = bool(use_agent_id_emb) and num_a > 0
        self.emb_q = nn.Embedding(num_q, factors)
        self.emb_llm = nn.Embedding(num_llm_ids, factors) if self.use_llm_id_emb else None
        self.emb_tool = nn.Embedding(num_tool_ids, factors) if self.use_tool_id_emb else None
        self.emb_agent = nn.Embedding(num_a, factors) if self.use_agent_id_emb else None
        self.alpha_llm = nn.Parameter(torch.tensor(alpha_llm, dtype=torch.float32))
        self.alpha_tool = nn.Parameter(torch.tensor(alpha_tool, dtype=torch.float32))
        self.alpha_agent = nn.Parameter(torch.tensor(alpha_agent, dtype=torch.float32))
        self.add_bias = add_bias
        if add_bias:
            self.bias_q = nn.Embedding(num_q, 1)
            if self.use_agent_bias:
                self.bias_a = nn.Embedding(num_a, 1)

        self.register_buffer("agent_llm_idx", agent_llm_idx.long())
        if self.use_tool_id_emb:
            if agent_tool_indices_padded is None or agent_tool_mask is None:
                raise ValueError(
                    "MF with use_tool_id_emb=True requires agent_tool_indices_padded and agent_tool_mask."
                )
            self.register_buffer("agent_tool_indices_padded", agent_tool_indices_padded.long())
            self.register_buffer("agent_tool_mask", agent_tool_mask.float())
        else:
            self.register_buffer("agent_tool_indices_padded", torch.zeros((num_a, 1), dtype=torch.long))
            self.register_buffer("agent_tool_mask", torch.zeros((num_a, 1), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_q.weight)
        if self.emb_llm is not None:
            nn.init.xavier_uniform_(self.emb_llm.weight)
        if self.emb_tool is not None:
            nn.init.xavier_uniform_(self.emb_tool.weight)
        if self.emb_agent is not None:
            nn.init.xavier_uniform_(self.emb_agent.weight)
        if self.add_bias:
            nn.init.zeros_(self.bias_q.weight)
            if hasattr(self, "bias_a"):
                nn.init.zeros_(self.bias_a.weight)

    def score_embeddings(
        self,
        q_vec: torch.Tensor,
        a_vec: torch.Tensor,
        qi: Optional[torch.Tensor] = None,
        ai: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s = (q_vec * a_vec).sum(dim=-1)
        if self.add_bias and qi is not None:
            s = s + self.bias_q(qi).squeeze(-1)
            if ai is not None and hasattr(self, "bias_a"):
                s = s + self.bias_a(ai).squeeze(-1)
        return s

    def _mean_embed_tools(self, a_idx: torch.LongTensor) -> torch.Tensor:
        if not self.use_tool_id_emb or self.emb_tool is None:
            return torch.zeros((a_idx.size(0), self.emb_q.embedding_dim), device=a_idx.device, dtype=self.emb_q.weight.dtype)
        tool_ids = self.agent_tool_indices_padded[a_idx]
        tool_mask = self.agent_tool_mask[a_idx].unsqueeze(-1)
        tool_emb = self.emb_tool(tool_ids)
        denom = tool_mask.sum(dim=1).clamp_min(1.0)
        return (tool_emb * tool_mask).sum(dim=1) / denom

    def agent_repr_batch(self, a_idx: torch.LongTensor) -> torch.Tensor:
        a_idx = a_idx.long()
        out = torch.zeros((a_idx.size(0), self.emb_q.embedding_dim), device=a_idx.device, dtype=self.emb_q.weight.dtype)
        if self.use_llm_id_emb and self.emb_llm is not None:
            out = out + self.alpha_llm * self.emb_llm(self.agent_llm_idx[a_idx].long())
        if self.use_tool_id_emb and self.emb_tool is not None:
            out = out + self.alpha_tool * self._mean_embed_tools(a_idx)
        if self.use_agent_id_emb and self.emb_agent is not None:
            out = out + self.alpha_agent * self.emb_agent(a_idx)
        return out

    def forward(
        self,
        q_idx: torch.LongTensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ensure long + same device
        q_idx = q_idx.long()
        pos_idx = pos_idx.long()
        neg_idx = neg_idx.long()

        qv = self.emb_q(q_idx)
        apv = self.agent_repr_batch(pos_idx)
        anv = self.agent_repr_batch(neg_idx)

        pos = self.score_embeddings(qv, apv, q_idx, pos_idx)
        neg = self.score_embeddings(qv, anv, q_idx, neg_idx)
        return pos, neg

    # ---- RecommenderBase ----
    def export_agent_embeddings(self, batch_size: Optional[int] = 4096) -> np.ndarray:
        with torch.no_grad():
            device = self.emb_q.weight.device
            if batch_size is None or batch_size >= self.num_a:
                a_idx = torch.arange(self.num_a, device=device, dtype=torch.long)
                return self.agent_repr_batch(a_idx).detach().cpu().numpy().astype(np.float32)

            chunks = []
            for start in range(0, self.num_a, batch_size):
                end = min(start + batch_size, self.num_a)
                a_idx = torch.arange(start, end, device=device, dtype=torch.long)
                chunks.append(self.agent_repr_batch(a_idx).detach().cpu().numpy().astype(np.float32))
            return np.concatenate(chunks, axis=0)

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        w = self.emb_q.weight.detach().cpu().numpy().astype(np.float32)
        return w[np.array(list(q_indices), dtype=np.int64)]

    def export_agent_bias(self) -> Optional[np.ndarray]:
        if hasattr(self, "bias_a"):
            return self.bias_a.weight.detach().cpu().numpy().squeeze(-1).astype(np.float32)
        return None

    def extra_state_dict(self) -> Dict[str, Any]:
        return {
            "add_bias": bool(self.add_bias),
            "use_agent_bias": bool(self.use_agent_bias),
            "use_llm_id_emb": bool(self.use_llm_id_emb),
            "use_tool_id_emb": bool(self.use_tool_id_emb),
            "use_agent_id_emb": bool(self.use_agent_id_emb),
        }
