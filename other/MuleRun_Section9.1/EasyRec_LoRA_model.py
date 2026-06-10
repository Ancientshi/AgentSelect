#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA pipeline:
- Easyrec encoder
- contrastive triplet-style forward
- encode() for bi-encoder inference
- stable save/load support for merged checkpoints
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
    RobertaPreTrainedModel,
)


class MLPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        return self.activation(self.dense(features))


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        if self.pooler_type == "avg":
            return (
                (last_hidden * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(-1).unsqueeze(-1)
            )
        if self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            return (
                ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(-1).unsqueeze(-1)
            )
        if self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            return (
                ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(-1).unsqueeze(-1)
            )
        raise NotImplementedError


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def default_model_args(config):
    return SimpleNamespace(
        temp=getattr(config, "temp", 0.05),
        pooler_type=getattr(config, "pooler_type", "cls"),
        do_mlm=getattr(config, "do_mlm", False),
        mlm_weight=getattr(config, "mlm_weight", 0.1),
        mlp_only_train=getattr(config, "mlp_only_train", False),
    )


class Easyrec(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs.get("model_args", default_model_args(config))
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mlp = MLPLayer(config)
        self.lm_head = RobertaLMHead(config)
        self.pooler_type = getattr(self.model_args, "pooler_type", "cls")
        self.pooler = Pooler(self.pooler_type)
        self.sim = Similarity(temp=getattr(self.model_args, "temp", 0.05))
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _encode_branch(
        self,
        *,
        input_ids,
        attention_mask,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled = self.pooler(attention_mask, outputs)
        if self.pooler_type == "cls":
            pooled = self.mlp(pooled)
        return outputs, pooled

    def forward(
        self,
        user_input_ids=None,
        user_attention_mask=None,
        pos_item_input_ids=None,
        pos_item_attention_mask=None,
        neg_item_input_ids=None,
        neg_item_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_attention_mask=None,
        mlm_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, user_pooler_output = self._encode_branch(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        _, pos_item_pooler_output = self._encode_branch(
            input_ids=pos_item_input_ids,
            attention_mask=pos_item_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        _, neg_item_pooler_output = self._encode_branch(
            input_ids=neg_item_input_ids,
            attention_mask=neg_item_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if dist.is_initialized() and self.training:
            user_list = [torch.zeros_like(user_pooler_output) for _ in range(dist.get_world_size())]
            pos_item_list = [torch.zeros_like(pos_item_pooler_output) for _ in range(dist.get_world_size())]
            neg_item_list = [torch.zeros_like(neg_item_pooler_output) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=user_list, tensor=user_pooler_output.contiguous())
            dist.all_gather(tensor_list=pos_item_list, tensor=pos_item_pooler_output.contiguous())
            dist.all_gather(tensor_list=neg_item_list, tensor=neg_item_pooler_output.contiguous())

            user_list[dist.get_rank()] = user_pooler_output
            pos_item_list[dist.get_rank()] = pos_item_pooler_output
            neg_item_list[dist.get_rank()] = neg_item_pooler_output

            user_pooler_output = torch.cat(user_list, dim=0)
            pos_item_pooler_output = torch.cat(pos_item_list, dim=0)
            neg_item_pooler_output = torch.cat(neg_item_list, dim=0)

        pos_sim = self.sim(user_pooler_output.unsqueeze(1), pos_item_pooler_output.unsqueeze(0))
        neg_sim = self.sim(user_pooler_output.unsqueeze(1), neg_item_pooler_output.unsqueeze(0))
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        labels = torch.arange(logits.size(0), dtype=torch.long, device=self.device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        if not return_dict:
            raise NotImplementedError

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooler_output = self.pooler(attention_mask, outputs)
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)
        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )
