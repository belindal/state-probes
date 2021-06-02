from typing import Iterable, List, Optional, Tuple

import torch
from transformers import BartConfig, T5Config
from transformers import BartTokenizerFast, T5TokenizerFast
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from torch import nn
import numpy as np
import random

from transformers import AdamW
from transformers.models.bart.modeling_bart import BartAttention as Attention
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers import PreTrainedModel
import os
import json
from tqdm import tqdm
from data.alchemy.utils import encodeState

import torch.nn.functional as F
import itertools
import sys
from localizer import LocalizerBase


def get_lang_model(arch, lm_save_path, pretrained=True, local_files_only=False, n_layers=None, device='cuda'):
    if arch == 'bart':
        model_class = BartForConditionalGeneration
        config_class = BartConfig
        model_fp = 'facebook/bart-base'
        tokenizer = BartTokenizerFast.from_pretrained(model_fp, local_files_only=local_files_only)
    elif arch == 't5':
        model_class = T5ForConditionalGeneration
        config_class = T5Config
        model_fp = 't5-base'
        tokenizer = T5TokenizerFast.from_pretrained(model_fp, local_files_only=local_files_only)
    else:
        raise NotImplementedError()

    if lm_save_path:
        print(f"Loading model from {lm_save_path}")
        model_dict = torch.load(lm_save_path, map_location=torch.device('cpu'))

    if n_layers is not None:
        assert not pretrained
    if not lm_save_path and pretrained:
        model = model_class.from_pretrained(model_fp, local_files_only=local_files_only)
    else:
        config = config_class.from_pretrained(model_fp, local_files_only=local_files_only)
        if n_layers is not None:
            if arch == 'bart':
                setattr(config, 'num_hidden_layers', n_layers)
                setattr(config, 'encoder_layers', n_layers)
                setattr(config, 'decoder_layers', n_layers)
            elif arch == 't5':
                setattr(config, 'num_layers', n_layers)
                setattr(config, 'num_decoder_layers', n_layers)
        model = model_class(config)
        if lm_save_path: model.load_state_dict(model_dict)
    encoder = model.get_encoder()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model, encoder, tokenizer


def get_state_encoder(arch, encoder=None, config=None, pretrained=True, freeze_params=True, local_files_only=False, n_layers=None, device='cuda'):
    # create/load world state encoder
    # (w/ same encoder as LM)
    print(f"Creating {arch}-style world state encoder")
    if not encoder:
        if arch == 'bart':
            if pretrained:
                state_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', local_files_only=local_files_only)
            else:
                config = BartConfig.from_pretrained('facebook/bart-base', local_files_only=local_files_only)
                setattr(config, 'num_hidden_layers', n_layers)
                setattr(config, 'encoder_layers', n_layers)
                setattr(config, 'decoder_layers', n_layers)
                state_model = BartForConditionalGeneration(config)
        elif arch =='t5':
            state_model = T5ForConditionalGeneration.from_pretrained('t5-base', local_files_only=local_files_only)
            if pretrained:
                state_model = T5ForConditionalGeneration.from_pretrained('t5-base', local_files_only=local_files_only)
            else:
                config = T5Config.from_pretrained('t5-base', local_files_only=local_files_only)
                setattr(config, 'num_layers', n_layers)
                setattr(config, 'num_decoder_layers', n_layers)
                state_model = T5ForConditionalGeneration(config)
        encoder = state_model.get_encoder()
    if arch == "mlp":
        input_dim = encodeState('alchemy', '1:', device).size(0)
        encoder = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.Sigmoid(),
            nn.Linear(config.d_model, config.d_model),
        )
    else: assert encoder

    if freeze_params:
        for p in encoder.parameters():
            p.requires_grad = False

    encoder.to(device)
    return encoder

def get_probe_model(probe_type, localizer_type, probe_attn_dim, arch, lang_model, probe_save_path, tgt_agg_method, encode_tgt_state=None, local_files_only=False, device='cuda'):
    load_probe = False
    if probe_save_path and os.path.exists(probe_save_path):
        load_probe = True
        print(f"Loading probe model from {probe_save_path}")
        probe_model_dict = torch.load(probe_save_path, map_location=torch.device('cpu'))

    if probe_type == 'decoder':
        if arch =='bart':
            if not load_probe:
                probe_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', local_files_only=local_files_only)
            else:
                config = BartConfig.from_pretrained('facebook/bart-base', local_files_only=local_files_only)
                probe_model = BartForConditionalGeneration(config)
        elif arch =='t5':
            if not load_probe:
                probe_model = T5ForConditionalGeneration.from_pretrained('t5-base', local_files_only=local_files_only)
            else:
                config = T5Config.from_pretrained('t5-base', local_files_only=local_files_only)
                probe_model = T5ForConditionalGeneration(config)
        else:
            raise NotImplementedError()
    elif probe_type.startswith('linear'):
        probe_model = nn.Linear(lang_model.config.d_model, lang_model.config.d_model)
    elif probe_type[1:].startswith('linear'):  # 3linear/2linear/nlinear
        probe_model = nn.Bilinear(lang_model.config.d_model, lang_model.config.d_model, int(probe_type[0]))
    elif probe_type.startswith('mlp'):
        probe_model = nn.Sequential(
            nn.Linear(lang_model.config.d_model, lang_model.config.d_model),
            nn.Sigmoid(),
            nn.Linear(lang_model.config.d_model, lang_model.config.d_model),
        )
    elif probe_type.startswith('lstm'):
        probe_model = nn.LSTM(lang_model.config.d_model, 512, batch_first=True, bidirectional=False)
    else:
        raise NotImplementedError()

    # create aggregation layer
    if localizer_type and localizer_type.endswith('_attn'):
        if localizer_type.startswith('lin_'):
            agg_layer = nn.Sequential(
                nn.Linear(lang_model.config.d_model, probe_attn_dim),
            )
        elif localizer_type.startswith('ffn_'):
            agg_layer = nn.Sequential(
                nn.Linear(lang_model.config.d_model, lang_model.config.d_model),
                nn.Sigmoid(),
                nn.Linear(lang_model.config.d_model, probe_attn_dim),
            )
        elif localizer_type.startswith('self_'):
            agg_layer = Attention(
                lang_model.config.d_model,
                lang_model.config.encoder_attention_heads, 
            )
        else: assert False
        probe_model.agg_layer = agg_layer
        probe_model.target_agg_layer = agg_layer
    if tgt_agg_method and tgt_agg_method.endswith('_attn'):
        if encode_tgt_state.split('.')[0] == "NL": input_dim = lang_model.config.d_model
        else: input_dim = encodeState('alchemy', '1:', device).size(0)
        if tgt_agg_method.startswith('lin_'):
            target_agg_layer = nn.Sequential(
                nn.Linear(input_dim, probe_attn_dim),
            )
        elif tgt_agg_method.startswith('ffn_'):
            target_agg_layer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid(),
                nn.Linear(input_dim, probe_attn_dim),
            )
        elif tgt_agg_method.startswith('self_'):
            target_agg_layer = Attention(
                input_dim,
                lang_model.config.encoder_attention_heads, 
            )
        else: assert False
        probe_model.target_agg_layer = target_agg_layer

    if load_probe:
        probe_model.load_state_dict(probe_model_dict)
        if probe_type == 'decoder':
            if arch == 't5': probe_model.encoder = lang_model.get_encoder()
            elif arch == 'bart': probe_model.model.encoder = lang_model.get_encoder()
    probe_model.to(device)
    return probe_model


class ProbeLanguageEncoder(nn.Module):
    def __init__(
        self,
        arch: str,
        probe_layer: int,
        base_lm: PreTrainedModel,
        probe_base_model: nn.Module,
        localizer: LocalizerBase,
    ):
        super().__init__()
        self.arch = arch
        self.probe_layer = probe_layer
        self.base_lm = base_lm
        self.base_encoder = self.base_lm.get_encoder()
        self.probe_base_model = probe_base_model
        self.localizer = localizer
    
    def forward(self, input_ids, attention_mask, offset_mapping=None, return_dict=False, output_attentions=False, output_hidden_states=False, localizer_key=None):
        '''
        `probe_outs (state)`: right *after* all commands in `inputs`, before `lang_tgts` command
        '''
        # forward language encoder
        encoder_outputs = self.base_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=output_attentions,
        )
        # [(bsz, seqlen, embed_dim) x (num_layers+1)]
        hidden_states = encoder_outputs.hidden_states
        probe_layer_hidden_states = hidden_states[self.probe_layer]
        
        if not output_hidden_states: hidden_states = None
        if output_attentions: attentions = encoder_outputs.attentions
        else: attentions = None

        # create encoding for lang side
        # (bsz, [cut_seqlen,] hidden_dim); (bsz, [cut_seqlen,])
        localized_encodings, localized_encodings_mask = self.localizer(probe_layer_hidden_states, input_ids, attention_mask, offset_mapping=offset_mapping, localizer_key=localizer_key)
        if return_dict:
            return BaseModelOutput(last_hidden_state=probe_layer_hidden_states, hidden_states=hidden_states, attentions=attentions)
        else:
            return localized_encodings, localized_encodings_mask, hidden_states, attentions


class ProbeBaseModel(PreTrainedModel):
    def __init__(
        self, arch, config, base_lm, base_state_model, probe_base_model, probe_layer, probe_type, localizer, state_localizer,
    ):
        super().__init__(config)
        self.arch = arch
        self.config = config
        self.base_lm = base_lm
        self.base_state_model = base_state_model
        self.probe_base_model = probe_base_model
        self.probe_type = probe_type
        self.localizer = localizer
        # localizes corresponding state
        self.state_localizer = state_localizer
        self.encoder = ProbeLanguageEncoder(arch, probe_layer, self.base_lm, self.probe_base_model, self.localizer)
    
    def get_encoder(self):
        return self.encoder


class ProbeLinearModel(ProbeBaseModel):
    def __init__(
        self, arch, config, base_lm, base_state_model, probe_base_model, probe_layer, probe_type, localizer, state_localizer,
    ):
        super().__init__(arch, config, base_lm, base_state_model, probe_base_model, probe_layer, probe_type, localizer, state_localizer)
    
    def forward(
        self, input_ids, attention_mask, offset_mapping=None, probe_outs=None,
        encoder_outputs=None, return_dict=False, output_attentions=False, output_hidden_states=False, localizer_key=None,
        **kwargs,
    ):
        extra_returns = {}
        if not encoder_outputs:
            probe_inputs, probe_inputs_mask, _, _ = self.encoder(
                input_ids, attention_mask, offset_mapping=offset_mapping, localizer_key=localizer_key,
            )
        else:
            probe_inputs = encoder_outputs
        
        # apply probe (transform on language encoding to state space)
        # (bsz, hidden_dim)
        if not self.probe_type[1:].startswith('linear'):
            transformed_encoded_reps = self.probe_base_model(probe_inputs)
            if len(transformed_encoded_reps.size()) == 3: transformed_encoded_reps = transformed_encoded_reps.sum(1)
            if self.probe_type == "lstm":
                # (bsz, seqlen, embeddim)
                transformed_encoded_reps = transformed_encoded_reps[0]

        # create encoding for states
        # (# total, seqlen, embeddim)
        all_vectors = probe_outs['all_states_encoding'].to(self.device)
        if len(all_vectors.size()) > 3:
            all_vectors = all_vectors.view(-1, all_vectors.size(-2), all_vectors.size(-1))
            probe_outs['all_states_input_ids'] = probe_outs['all_states_input_ids'].view(-1, probe_outs['all_states_input_ids'].size(-1))
            probe_outs['all_states_attn_mask'] = probe_outs['all_states_attn_mask'].view(-1, probe_outs['all_states_attn_mask'].size(-1))
        # (# total, 1, embeddim)
        all_vectors, all_vectors_mask = self.state_localizer(all_vectors, probe_outs['all_states_input_ids'], probe_outs['all_states_attn_mask'])
        if all_vectors.size(1) == 1:
            # (# total, embeddim)
            all_vectors = all_vectors.squeeze(1)
            all_vectors_mask = all_vectors_mask.squeeze(1)

        if self.probe_type[1:] == 'linear_classify':
            bs, numnegs, embeddim = probe_outs['all_states_encoding'].size(0), probe_outs['all_states_encoding'].size(1), all_vectors.size(-1)
            # n-way classification
            # (bs*c, #negs, embeddim)
            all_vectors = all_vectors.view(-1, numnegs, embeddim)
            all_vectors_mask = all_vectors_mask.view(-1, numnegs)
            probe_outs['all_states_attn_mask'] = probe_outs['all_states_attn_mask'].view(-1, numnegs)
            if all_vectors.size(0) == 1:
                all_vectors = all_vectors.repeat(bs,1,1)
                all_vectors_mask = all_vectors_mask.repeat(bs,1)
            # (bs, #negs, n)
            similarity_scores = self.probe_base_model(probe_inputs.repeat(1,numnegs,1), all_vectors)
            if 'labels' in probe_outs:
                label_mask = probe_outs['labels'] != -1
                assert (label_mask == all_vectors_mask).all()
                probe_loss = F.cross_entropy(similarity_scores[all_vectors_mask], probe_outs['labels'][label_mask])
            else:
                probe_loss = None
            extra_returns['similarity'] = similarity_scores
        else:
            # (bsz, # total examples)
            similarity_scores = torch.matmul(transformed_encoded_reps, all_vectors.t())
            probe_loss = F.cross_entropy(similarity_scores, probe_outs['labels'])
            # batchwise negatives
            extra_returns["similarity"] = similarity_scores

        return {"loss": probe_loss, **extra_returns}
    

class ProbeConditionalGenerationModel(ProbeBaseModel):
    def __init__(
        self, arch, config, base_lm, base_state_model, probe_base_model, probe_layer, localizer, state_localizer,
    ):
        super().__init__(arch, config, base_lm, base_state_model, probe_base_model, probe_layer, localizer, state_localizer)
    
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        return self.probe_base_model.prepare_inputs_for_generation(inputs, **kwargs)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.probe_base_model.prepare_decoder_input_ids_from_labels(labels)

    def _reorder_cache(self, past, beam_idx):
        return self.probe_base_model._reorder_cache(past, beam_idx)

    def forward(
        self, input_ids, attention_mask, offset_mapping=None, probe_outs=None,
        encoder_outputs=None, return_dict=False, output_attentions=False, output_hidden_states=False, localizer_key=None,
        labels=None, decoder_input_ids=None, **kwargs,
    ):
        if not encoder_outputs:
            probe_inputs, probe_inputs_mask, _, _ = self.encoder(
                input_ids, attention_mask, offset_mapping=offset_mapping, localizer_key=localizer_key,
            )
        else:
            probe_inputs = encoder_outputs.last_hidden_state

        assert self.probe_type == 'decoder'
        assert len(probe_inputs.size()) > 2
        if probe_outs:
            # override `labels` and `decoder_input_ids`
            labels = probe_outs['input_ids']
            decoder_input_ids = probe_outs['input_ids']
        all_returns = self.probe_base_model(input_ids=None, encoder_outputs=(probe_inputs,), decoder_input_ids=decoder_input_ids, labels=labels, **kwargs)
        return ModelOutput(
            **all_returns,
            decoder_inputs=probe_inputs,
            last_hidden_state=probe_inputs,
        )


def encode_target_states(state_model, dataset, tokenizer, encode_init_state, probe_model, args, all_state_targets=None):
    """
    Specify either dataset or all_state_targets
    """
    # get all examples in the dataset (input + attention mask)
    if all_state_targets is None:
        maxseqlen = 128
        all_state_input_ids = []
        all_state_attn_mask = []
        all_agg_sentence_rep = []
        all_agg_sentence_rep_mask = []
        for (inputs, lang_tgts, state_tgts, raw_state_targets, init_states) in convert_to_transformer_batches(
            args, dataset, tokenizer, args.eval_batchsize, include_init_state=encode_init_state, no_context=args.no_context,
            append_last_state_to_context=args.append_last_state_to_context, domain="alchemy", state_targets_type=args.probe_target,
        ):
            '''
            model forward
            '''
            all_state_input_ids.append(F.pad(state_tgts['input_ids'], (0, maxseqlen - state_tgts['input_ids'].size(1), 0, 0), value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)))
            all_state_attn_mask.append(F.pad(state_tgts['attention_mask'], (0, maxseqlen - state_tgts['attention_mask'].size(1), 0, 0), value=0))
            # encode everything
            # (bs, seqlen, embeddim)
            agg_sentence_rep = state_model(input_ids=state_tgts['input_ids'], attention_mask=state_tgts['attention_mask'])[0]
            all_agg_sentence_rep.append(agg_sentence_rep)
        all_state_input_ids = torch.cat(all_state_input_ids, dim=0)
        all_state_attn_mask = torch.cat(all_state_attn_mask, dim=0)
        all_agg_sentence_rep = torch.cat(all_agg_sentence_rep, dim=0)
    else:
        all_state_input_ids = all_state_targets['input_ids']
        all_state_attn_mask = all_state_targets['attention_mask']
        '''
        model forward
        '''
        # encode everything
        # (bs, seqlen, embeddim)
        all_agg_sentence_rep = state_model(input_ids=all_state_targets['input_ids'], attention_mask=all_state_targets['attention_mask'])[0]
        
    # build index
    all_state_index = None
    return all_state_input_ids, all_state_attn_mask, all_agg_sentence_rep, all_state_index

