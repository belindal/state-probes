from typing import Iterable, List, Optional, Tuple

import torch
from data.alchemy.parse_alchemy import colors
from data.alchemy.utils import int_to_word
import sys
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from torch import nn
from . import LocalizerBase
import regex as re


class SconeLocalizer(LocalizerBase):
    def __init__(
        self,
        agg_layer: nn.Module,
        agg_method: str,
        attn_dim: str,
        localizer_type: str,
        max_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        device: int,
    ):
        """
        Base class for localizer
        """
        super().__init__(agg_layer, agg_method, attn_dim, localizer_type, max_tokens, tokenizer, device)
        
        if self.localizer_type == "init_state": 
            self.localizer = self.localize_init_state
        
        elif self.localizer_type.startswith('single_beaker'):
            self.localizer = self.localize_single_beaker_init_decl
        
        elif self.localizer_type == 'all':
            self.localizer = self.identity_localizer
        
        else:
            raise NotImplementedError()

    
    def forward(self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, offset_mapping: torch.Tensor = None, localizer_key: Optional[str] = None, *args, **kwargs,):
        # encoded_reps (bsz x seqlen x hidden_dim): state or lang encodings
        # input_ids: bsz x seqlen
        localized_encodings, localized_encodings_mask = self.localizer(encoded_reps, input_ids, attention_mask, offset_mapping=offset_mapping, beaker_idx=localizer_key)
        return self.aggregate_token_encodings(localized_encodings, localized_encodings_mask)

    def localize_init_state(
        self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs,
    ):
        encoded_reps_mask = attention_mask.clone()
        split_pos = (input_ids == self.tokenizer.convert_tokens_to_ids('.')).nonzero(as_tuple=False)
        assert (split_pos[:,0] == torch.arange(split_pos.size(0)).to(self.device)).all()
        assert (split_pos[:,1] == split_pos[0,1]).all()
        encoded_reps = encoded_reps[:, :split_pos[0,1], :]
        encoded_reps_mask = encoded_reps_mask[:, :split_pos[0,1]]
        return encoded_reps, encoded_reps_mask

    def localize_single_beaker_init_decl(
        self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, offset_mapping: torch.Tensor, beaker_idx: int = -1, *args, **kwargs,
    ):
        encoded_reps_mask = attention_mask.clone()
        # unalign for control tasks
        state_offset = 0
        token_offset = None
        localizer_type = self.localizer_type.replace('single_beaker_', '')
        beaker2input_align_type = localizer_type.split('.')[0]
        localizer_type = localizer_type.replace(beaker2input_align_type, "")
        B, S = input_ids.size(0), input_ids.size(1)
        
        if localizer_type.startswith(".R"):
            localizer_type = localizer_type.replace(".R", "")
            token_offset = int(localizer_type.split('.')[0])
            localizer_type = localizer_type[len(str(token_offset)):]
        if localizer_type.startswith(".offset"):
            localizer_type = localizer_type.replace(".offset", "")
            state_offset = int(localizer_type.split('.')[0])
            localizer_type = localizer_type[len(str(state_offset)):]
        # get lang representation at "the first beaker"
        if beaker_idx == 'original_text' or beaker_idx == 'full_state':
            beaker_idx = 0
        encoded_pos_to_get = (beaker_idx + state_offset) % len(int_to_word)
        # match encoded representation at `beaker_idx + state_offset` with gold token at `beaker_idx`
        init_segment_split = f"."
        # make mention_portion_split_pos_mask: True in desired portion, False elsewhere (i.e. for `init`, before '.' and False afterwards)
        mention_portion_split_pos_mask = input_ids == torch.tensor(self.tokenizer.encode(init_segment_split)[1:-1]).to(self.device)
        init_split_pos = torch.arange(mention_portion_split_pos_mask.size(1)).to(self.device).unsqueeze(0).expand_as(mention_portion_split_pos_mask).masked_fill(~mention_portion_split_pos_mask, sys.maxsize-1).min(-1)[0]
        mention_portion_split_pos_mask[:,:] = False
        mention_portion_split_pos_mask[:,:init_split_pos.min()] = True
        for i in range(init_split_pos.min(), init_split_pos.max()):
            # exclude the period (if '.' at position > i, set i to True)
            mention_portion_split_pos_mask[init_split_pos > i, i] = True
        beaker_segment_split = f","
        beaker_split_pos_mask = input_ids == torch.tensor(self.tokenizer.encode(beaker_segment_split)[1:-1]).to(self.device)
        beaker_split_pos_mask &= mention_portion_split_pos_mask  # only in the init portion
        # Get beaker positions
        # (bsz, seqlen)
        beaker_split_pos = torch.arange(beaker_split_pos_mask.size(1)).to(self.device).unsqueeze(0).expand_as(
            beaker_split_pos_mask).masked_fill(~beaker_split_pos_mask, sys.maxsize).sort(-1)[0]
        begin_token = beaker_split_pos[:,encoded_pos_to_get-1] if encoded_pos_to_get > 0 else torch.tensor([0]).to(self.device)
        if self.tokenizer.bos_token_id is not None or encoded_pos_to_get > 0: begin_token += 1  # skip bos or after some comma
        end_token = beaker_split_pos[:,encoded_pos_to_get] if encoded_pos_to_get < 6 else init_split_pos
        # mask for beaker position
        beaker_pos_mask = beaker_split_pos_mask.clone()
        beaker_pos_mask[:,:] = False
        for i in range(begin_token.min(), end_token.max()+1):
            # inclusive begin, inclusive end (include the `,`)
            beaker_pos_mask[(begin_token <= i) & (i <= end_token), i] = True
        if beaker2input_align_type in ['init_color', 'init_amount', 'init_pos', 'init_verb', 'init_beaker', 'init_article', 'init_end_punct']:
            if beaker2input_align_type == 'init_color':
                all_toks = [f'{c}' for c in colors] + ['empty']
            elif beaker2input_align_type == 'init_amount':
                all_toks = ["1", "2", "3", "4", "empty"]
            elif beaker2input_align_type == 'init_pos':
                all_toks = [f'{p}' for p in [int_to_word[encoded_pos_to_get]]]
            elif beaker2input_align_type == 'init_beaker':
                all_toks = ["beaker"]
            elif beaker2input_align_type == 'init_verb':
                all_toks = ["is", "has"]
            elif beaker2input_align_type == 'init_article':
                all_toks = ["the"]
            elif beaker2input_align_type == 'init_end_punct':
                all_toks = [",", "."]
            
            toks_mask = torch.zeros(input_ids.size()).bool().to(self.device)
            for idx in range(B):
                ex_offset_mapping = offset_mapping[idx][(input_ids[idx] != self.tokenizer.pad_token_id) & (input_ids[idx] != self.tokenizer.eos_token_id)]
                for t in all_toks:
                    if t == '.': t = '\.'
                    for mb in re.finditer(t, self.tokenizer.decode(input_ids[idx])):
                        tok_start_incl = (ex_offset_mapping[:,0] <= mb.start()).nonzero(as_tuple=False).max()
                        tok_end_incl = (offset_mapping[idx][:,1] >= mb.end()).nonzero(as_tuple=False).min()
                        if tok_end_incl < tok_start_incl:
                            # non-monotonic
                            tok_start_incl = tok_end_incl
                        if token_offset is None:
                            tok_pos = tok_start_incl
                        else:
                            if tok_start_incl + token_offset > tok_end_incl: token_offset = 0
                            tok_pos = tok_start_incl + token_offset
                        assert tok_pos <= tok_end_incl
                        toks_mask[idx, tok_pos:tok_pos+1] = True

            # (bsz, 1, embeddim)
            encoded_reps = encoded_reps[beaker_pos_mask & toks_mask].unsqueeze(1)
            # (bsz, 1)
            encoded_reps_mask = encoded_reps_mask[beaker_pos_mask & toks_mask].unsqueeze(-1)
        elif token_offset is not None:
            # init_full, token_offset specified
            # only get 1 token, specified by `token_offset`
            batch_range = torch.arange(encoded_reps.size(0)).to(self.device)
            # (bsz, 1, embeddim)
            encoded_reps = encoded_reps[batch_range, begin_token + token_offset, :].unsqueeze(1)
            # (bsz, 1)
            encoded_reps_mask = encoded_reps_mask[batch_range, begin_token + token_offset].unsqueeze(-1)
        else:
            # set encoded_reps
            encoded_reps = encoded_reps.clone()
            encoded_reps[~beaker_pos_mask] = 0
            encoded_reps_mask[~beaker_pos_mask] = 0
        return encoded_reps, encoded_reps_mask
