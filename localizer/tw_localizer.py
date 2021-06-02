from typing import Iterable, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase
from torch import nn
from . import LocalizerBase


class TWLocalizer(LocalizerBase):
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
        super().__init__(agg_layer, agg_method, attn_dim, localizer_type, max_tokens, tokenizer)

        if 'belief_facts' in self.localizer_type:
            self.localizer = self.localize_entities
        
        elif self.localizer_type == 'all':
            self.localizer = self.identity_localizer

        else:
            raise NotImplementedError()
    
    def forward(self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, offset_mapping: torch.Tensor = None, localizer_key: Optional[List[str]] = None):
        # inputs['entities']
        S = encoded_reps.size(1)
        # (B, S) -- indices of `>` and special tokens in input_ids
        command_bounds = (
            torch.arange(S, device=self.device)
            .unsqueeze(0)
            .expand_as(input_ids)
            .masked_fill(
                (input_ids != self.tokenizer.convert_tokens_to_ids('>')) &
                (input_ids != self.tokenizer.eos_token_id) &
                (input_ids != self.tokenizer.bos_token_id) &
                (input_ids != self.tokenizer.pad_token_id)
            , S + 1)
        )
        localized_encodings, localized_encodings_mask = self.localizer(encoded_reps, input_ids, attention_mask.clone(), command_bounds, offset_mapping=offset_mapping, mentions=localizer_key)
        return self.aggregate_token_encodings(localized_encodings, localized_encodings_mask)
    

    def localize_entities(self, encoded_reps: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, command_bounds: torch.Tensor, offset_mapping: torch.Tensor, mentions: List[List[int]]):
        occurrence = self.localizer_type.split('_')[-1]  # all, first, last
        # (B, max_range_len); (B, max_range_len)
        selected_indices, selection_mask = self.get_entity_indices(input_ids, offset_mapping, mentions, command_bounds, occurrence)
        # (B, max_range_len, E); (B, max_range_len); (B, max_range_len)
        encoded_reps, encoded_reps_mask, selected_input_ids = self.select_batch_indices(
            selected_indices, selection_mask, encoded_reps, attention_mask, input_ids)
        return encoded_reps, encoded_reps_mask


    def select_batch_indices(self, selected_indices, selection_mask, encoded_reps, encoded_reps_mask, input_ids):
        """
        Inputs:
            selected_indices (B x max_range_len): indices to select
            selection_mask (B x max_range_len): mask on selected_indices -- True where *not* to mask
            encoded_reps (B x S x E)
            encoded_reps_mask (B x S)
            input_ids (B x S)
        """
        B, S, E = encoded_reps.size(0), encoded_reps.size(1), encoded_reps.size(2)
        selected_indices[~selection_mask] = 0
        # (B, max_range_len, E)
        encoded_reps = encoded_reps.gather(1, selected_indices.unsqueeze(-1).expand(-1, -1, E))
        # (B, max_range_len)
        encoded_reps_mask = encoded_reps_mask.gather(1, selected_indices) & selection_mask
        selected_input_ids = input_ids.gather(1, selected_indices)
        return encoded_reps, encoded_reps_mask, selected_input_ids


    def get_entity_indices(self, input_ids, offset_mapping, mentions, command_bounds, occurrence):
        # get actions associated with entity(s)
        B, S = input_ids.size(0), input_ids.size(1)
        selected_index_mask = []
        for i in range(B):
            all_entity_input_pos_mask_agg = None
            selected_index_mask.append(torch.zeros(S).bool())
            # context_tokens = self.tokenizer.decode(input_ids[i])
            ex_offset_mapping = offset_mapping[i][(input_ids[i] != self.tokenizer.pad_token_id) & (input_ids[i] != self.tokenizer.eos_token_id)]
            # across mention set
            for mention_bounds in mentions[i]:
                if mention_bounds is None: continue
                # Order mention bounds
                tgt_mention_bounds = []
                for mb in mention_bounds:
                    mb_start = mb.span()[0]
                    if mb.string.startswith(' ') or mb.string.startswith('\n'): mb_start += 1
                    tgt_mention_bounds.append([mb_start, mb.span()[1]-1])
                tgt_mention_bounds = sorted(tgt_mention_bounds, key=lambda x: x[0])
                if occurrence == 'first':
                    # (B, 1)
                    tgt_mention_bounds = [tgt_mention_bounds[0]]
                elif occurrence == 'last':
                    # (B, 1) -- take just before the paddings
                    tgt_mention_bounds = [tgt_mention_bounds[-1]]
                else: assert occurrence == 'all'
                entity_input_pos_mask_agg = torch.zeros(input_ids.size(1)).bool()
                for mb in tgt_mention_bounds:
                    tok_start_incl = (ex_offset_mapping[:,0]<=mb[0]).nonzero(as_tuple=False).max()
                    tok_end_incl = (offset_mapping[i][:,1]>=mb[1]).nonzero(as_tuple=False).min()
                    entity_input_pos_mask_agg[tok_start_incl:tok_end_incl+1] = True
                # only `True` at first/last occurrence of all entities in set
                if all_entity_input_pos_mask_agg is None: all_entity_input_pos_mask_agg = entity_input_pos_mask_agg
                else: all_entity_input_pos_mask_agg |= entity_input_pos_mask_agg
            selected_index_mask[i] = all_entity_input_pos_mask_agg
        selected_index_mask = torch.stack(selected_index_mask).to(self.device)
        # turn to indices
        selected_indices, selection_mask = self.convert_batch_mask_to_indices(selected_index_mask)
        return selected_indices, selection_mask


    def convert_batch_mask_to_indices(self, mask):
        # mask: bsz x seqlen
        # convert to indices where `True`
        B, S = mask.size(0), mask.size(1)
        indices = (mask * torch.arange(S).unsqueeze(0).to(mask.device)).masked_fill(~mask, S+1).sort().values
        indices_mask = indices != S+1
        # crop to max
        longest_sequence = indices_mask.sum(-1).max()
        indices = indices[:,:longest_sequence]
        indices_mask = indices_mask[:,:longest_sequence]
        return indices, indices_mask
