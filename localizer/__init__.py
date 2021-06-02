from typing import Iterable, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase
from torch import nn


class LocalizerBase(nn.Module):
    def __init__(
        self,
        agg_layer: nn.Module,
        agg_method: str,
        attn_dim: str,
        localizer_type: str,
        max_tokens: int,
        tokenizer: PreTrainedTokenizerBase, 
        device: str = 'cuda',
    ):
        """
        Base class for localizer
        """
        super().__init__()
        self.agg_layer = agg_layer
        self.agg_method = agg_method
        self.attn_dim = attn_dim
        self.max_tokens = max_tokens
        self.localizer_type = localizer_type
        self.tokenizer = tokenizer
        self.device = device
    
    def forward(self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.identity_localizer(encoded_reps, input_tokens, input_ids_key, input_mask_key)
    
    def identity_localizer(self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs):
        # returns all tokens
        return encoded_reps, attention_mask

    def aggregate_token_encodings(self, encoded_reps, encoded_reps_mask):
        # then apply aggregation
        # (bsz x seqlen x hidden_dim) --> (bsz x hidden_dim)
        # TODO account for mask tokens in each of these!!
        if self.agg_method == "sum":
            # (bsz x seqlen x 1)
            encoded_reps_mask = encoded_reps_mask.unsqueeze(-1)
            # (bsz x 1 x hidden_dim)
            encoded_reps = (encoded_reps * encoded_reps_mask).sum(1, keepdim=True)
            encoded_reps_mask = encoded_reps_mask.sum(1, keepdim=True) != 0
        elif self.agg_method == "avg":
            # (bsz x seqlen x 1)
            encoded_reps_mask = encoded_reps_mask.unsqueeze(-1)
            # (bsz x 1 x hidden_dim)
            encoded_reps = (encoded_reps * encoded_reps_mask).sum(1, keepdim=True) / encoded_reps_mask.sum(1, keepdim=True) # exclude mask
            encoded_reps_mask = encoded_reps_mask.sum(1) != 0
            encoded_reps[~encoded_reps_mask] = 0
        elif self.agg_method == 'first':
            # (bsz x 1 x hidden_dim)
            encoded_reps = encoded_reps[:,0,:].unsqueeze(1)
            # (bsz x 1)
            encoded_reps_mask = encoded_reps_mask[:,0].unsqueeze(-1)
        elif self.agg_method == 'last':
            last_idxs = (encoded_reps_mask * torch.arange(encoded_reps_mask.size(1)).unsqueeze(0).to(self.device)).max(1)[0]
            # (bsz x 1 x hidden_dim)
            encoded_reps = encoded_reps[torch.arange(encoded_reps.size(0)).to(self.device), last_idxs, :].unsqueeze(1)
            # (bsz x 1)
            encoded_reps_mask = encoded_reps_mask[torch.arange(encoded_reps.size(0)).to(self.device), last_idxs].unsqueeze(-1)
        elif self.agg_method and self.agg_method.endswith('_attn'):
            if self.agg_method.startswith('lin_') or self.agg_method.startswith('ffn_'):
                # (bsz x seqlen x compress_dim)
                attn_weights = self.agg_layer(encoded_reps)#, attention_mask)
                attn_weights = attn_weights.masked_fill((~encoded_reps_mask.bool()).unsqueeze(-1), float("-inf"))
                attn_weights = F.softmax(attn_weights, 1)
                # (bsz x compress_dim x hidden_dim) = (bsz x seqlen x compress_dim)^T * (bsz x seqlen x hidden_dim)
                encoded_reps = torch.bmm(attn_weights.permute(0,2,1), encoded_reps)
            elif self.agg_method.startswith('self_'):
                # (seqlen, bsz, hidden_dim)
                encoded_reps = encoded_reps.permute(1,0,2)
                # (seqlen x bsz x hidden_dim)
                attn_output, _ = self.agg_layer(encoded_reps, encoded_reps, key_padding_mask=(~encoded_reps_mask.bool()))
                # (bsz x compress_dim x hidden_dim)
                encoded_reps = attn_output[0:self.attn_dim,:,:].permute(1,0,2)
            else: assert False
            encoded_reps_mask = torch.ones(encoded_reps.size()[:len(encoded_reps.size())-1]).to(encoded_reps.device)
        else: assert not self.agg_method

        # cut to max_tokens
        # (bsz x seqlen x hidden_dim) --> (bsz x max_tokens x hidden_dim)
        if self.max_tokens and encoded_reps.size(1) > self.max_tokens:
            encoded_reps = encoded_reps[:,:self.max_tokens,:]
            encoded_reps_mask = encoded_reps_mask[:,:self.max_tokens]
        
        return encoded_reps, encoded_reps_mask