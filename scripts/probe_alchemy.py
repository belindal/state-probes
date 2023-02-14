import torch
from transformers.modeling_outputs import ModelOutput
from torch import nn
import numpy as np
import random
from copy import deepcopy

from transformers import AdamW
from data.alchemy.parseScone import loadData
import os
import json
from tqdm import tqdm

from data.alchemy.alchemy_artificial_generator import execute
from data.alchemy.parse_alchemy import (
    consistencyCheck, parse_utt_with_world, parse_world,
)
from metrics.alchemy_metrics import get_state_similarity, check_val_consistency
from data.alchemy.utils import (
    int_to_word, gen_all_beaker_states, get_matching_state_labels,
    translate_states_to_nl, translate_nl_to_states,
)
from data.alchemy.scone_dataloader import convert_to_transformer_batches
from localizer.scone_localizer import SconeLocalizer
import torch.nn.functional as F
import itertools
import Levenshtein

from probe_models import (
    ProbeLinearModel, ProbeConditionalGenerationModel, ProbeLanguageEncoder, encode_target_states,
    get_probe_model, get_state_encoder, get_lang_model,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--control_input', action='store_true')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--eval_batchsize', type=int, default=128)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--arch', type=str, default='bart', choices=['t5', 'bart', 'bert'])
parser.add_argument('--encode_tgt_state', type=str, default=False, choices=[False, 'raw.mlp', 'NL.bart', 'NL.t5'], help="how to encode the state before probing")
parser.add_argument('--tgt_agg_method', type=str, choices=['sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default='avg', help="how to aggregate across tokens of target, if `encode_tgt_state` is set True")
parser.add_argument('--probe_type', type=str, choices=['linear', 'mlp', 'lstm', 'decoder'], default='decoder')
parser.add_argument('--encode_init_state', type=str, default=False, choices=[False, 'raw', 'NL'])
parser.add_argument('--seed', type=int, default=45)
parser.add_argument('--no_context', action='store_true')
parser.add_argument('--append_last_state_to_context', action='store_true')
parser.add_argument('--nonsynthetic', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--no_pretrain', action='store_true')
parser.add_argument('--lm_save_path', type=str, default=None, help="load existing LM checkpoint (if any)")
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--probe_save_path', type=str, default=None, help="load existing state model checkpoint (if any)")
parser.add_argument('--probe_layer', type=int, default=-1, help="which layer of the model to probe")
parser.add_argument('--probe_target', type=str, choices=['text'] + [f'{target}.{text_type}' for target, text_type in itertools.product([
    'state', 'init_state', 'single_beaker_init', 'single_beaker_final',
], ['NL', 'raw'])], default="state.NL", help="what to probe for")
parser.add_argument('--probe_agg_method', type=str, choices=[None, 'sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default=None, help="how to aggregate across tokens")
parser.add_argument('--probe_attn_dim', type=int, default=None, help="what dimensions to compress sequence tokens to")
single_beaker_opts = ['single_beaker_init', 'single_beaker_init_color', 'single_beaker_final', 'single_beaker_all'] + [
    f'single_beaker_init_{att}'
    for att in ['amount', 'full', 'verb', 'article', 'end_punct', 'pos.R0', 'pos.R1', 'pos.R2', 'beaker.R0', 'beaker.R1', 'beaker.R2']
]
parser.add_argument('--localizer_type', type=str, default='all',
    choices=['all', 'init_state'] + single_beaker_opts + [f'{opt}.offset{i}' for opt in single_beaker_opts for i in range(7)] + [
        f'single_beaker_{occurrence}{token_offset}{offset}' for occurrence in ["init", "init_full"] for offset in [""] + [f".offset{i}" for i in range(7)] for token_offset in [""] + [f".R{j}" for j in range(-7, 8)]
    ],
    help="which encoded tokens of the input to probe; `offset` gives how much ahead the encoded representation should be; `R` gives the offset in tokens, relative to 'beaker'")
parser.add_argument('--probe_max_tokens', type=int, default=None, help="how many tokens (max) to feed into probe, set None to use all tokens")
parser.add_argument('--eval_only', action='store_true')
args = parser.parse_args()

BATCHSIZE = args.batchsize
EVAL_BATCHSIZE = args.eval_batchsize
arch = args.arch
encode_tgt_state = args.encode_tgt_state
probe_type = args.probe_type
probe_layer = args.probe_layer
encode_init_state = args.encode_init_state
lm_save_path = args.lm_save_path
probe_save_path = args.probe_save_path
pretrained = not args.no_pretrain
probe_target = args.probe_target
localizer_type = args.localizer_type
probe_agg_method = args.probe_agg_method
tgt_agg_method = args.tgt_agg_method
probe_attn_dim = args.probe_attn_dim
probe_max_tokens = args.probe_max_tokens
lr = args.lr

if probe_target != 'text': similarity_fn = lambda x,y,z: get_state_similarity(x,y,z,probe_target)
else: similarity_fn = lambda x,y,z: Levenshtein.ratio(x,y)

# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# load (language) model
model, encoder, tokenizer = get_lang_model(arch, lm_save_path, pretrained, device=args.device)

# create/load world state encoder
# (w/ same encoder as LM)
state_model = None
if encode_tgt_state:
    state_model = get_state_encoder(
        encode_tgt_state.split('.')[-1], encoder, config=model.config, pretrained=pretrained, freeze_params=(encode_tgt_state.split('.')[1] != 'raw'), device=args.device
    )

# create save path
save_state = True
if lm_save_path:
    output_dir = f"{f'encoded_{encode_tgt_state}' if encode_tgt_state else ''}{probe_type}" + \
    f"_{localizer_type}{'.control_inp' if args.control_input else ''}_" + \
    f"{probe_agg_method}{probe_attn_dim if probe_agg_method and probe_agg_method.endswith('_attn') else ''}{probe_max_tokens if probe_max_tokens else ''}{tgt_agg_method if encode_tgt_state else ''}" + \
    f"_{lm_save_path.split('.')[0].split('/')[-1]}_l{probe_layer}{'_' + probe_target if probe_target != 'state' else ''}_{'real' if args.nonsynthetic else 'synth'}"
else:
    output_dir = f"{'nopt_' if not pretrained else ''}noft_{f'encoded_{encode_tgt_state}' if encode_tgt_state else ''}{probe_type}" + \
    f"_{localizer_type}{'.control_inp' if args.control_input else ''}_" + \
    f"{probe_agg_method}{probe_attn_dim if probe_agg_method and probe_agg_method.endswith('_attn') else ''}{probe_max_tokens if probe_max_tokens else ''}{tgt_agg_method if encode_tgt_state else ''}" + \
    f"_{arch}_initstate_{encode_init_state}_l{probe_layer}{'_' + probe_target if probe_target != 'state' else ''}_{'real' if args.nonsynthetic else 'synth'}"
output_json_fn = os.path.join("probe_models_alchemy", f"{output_dir}.jsonl")
if not os.path.exists(os.path.join("probe_models_alchemy", f"{'/'.join(output_dir.split('/')[:-1])}")):
    os.makedirs(os.path.join("probe_models_alchemy", f"{'/'.join(output_dir.split('/')[:-1])}"), exist_ok=True)
print(f"Saving to {output_json_fn}")
if not probe_save_path:
    probe_save_path = output_json_fn.replace(".jsonl", ".pState")
print(f"Saving probe checkpoints to {probe_save_path}")

# create probe model
probe_model = get_probe_model(probe_type, probe_agg_method, probe_attn_dim, arch, model, probe_save_path, args.tgt_agg_method, encode_tgt_state=args.encode_tgt_state, device=args.device)

# load optimizer
all_parameters = list(probe_model.parameters())
optimizer = AdamW(list(p for p in all_parameters if p.requires_grad), lr=lr)

localizer = SconeLocalizer(getattr(probe_model, 'agg_layer', None), args.probe_agg_method, args.probe_attn_dim, args.localizer_type, args.probe_max_tokens, tokenizer, args.device)
state_localizer = SconeLocalizer(getattr(probe_model, 'target_agg_layer', None), args.tgt_agg_method, args.probe_attn_dim, 'all', args.probe_max_tokens, tokenizer, args.device)
if args.encode_tgt_state:
    assert args.probe_type != 'decoder'
    full_joint_model = ProbeLinearModel(
        args.arch, getattr(probe_model, 'config', getattr(model, 'config', None)),
        model, state_model, probe_model, args.probe_layer, args.probe_type,
        localizer, state_localizer,
    )
else:
    assert args.probe_type == 'decoder'
    full_joint_model = ProbeConditionalGenerationModel(
        args.arch, getattr(probe_model, 'config', getattr(model, 'config', None)),
        model, state_model, probe_model, args.probe_layer, args.probe_type,
        localizer, state_localizer,
    )

# load data
print("Loading data")
dataset, lang_v, state_v = loadData(split="train", kind="alchemy", synthetic=(not args.nonsynthetic))
dev_dataset, lang_v_dev, state_v_dev = loadData(split="dev", kind="alchemy", synthetic=(not args.nonsynthetic))
best_val_loss = 10**10
best_epoch = -1
all_train_states = [" ".join(state) for _, _, state in [x for d in dataset for x in d.all_pairs()]]
all_dev_states = [" ".join(state) for _, _, state in [x for d in dev_dataset for x in d.all_pairs()]]
random.shuffle(all_dev_states)
all_train_inps = [" ".join(inps) for inps, _, _ in [x for d in dataset for x in d.all_pairs()]]
all_dev_inps = [" ".join(inps) for inps, _, _ in [x for d in dev_dataset for x in d.all_pairs()]]


#dev loss
model.eval()
probe_model.eval()
if encode_tgt_state:
    encoding = encode_tgt_state.split('.')[0]
    if state_model: state_model.eval()
    print("Getting vectors for all possible beaker states")
    with torch.no_grad():
        if 'single_beaker_init' in args.probe_target or 'single_beaker_final' in args.probe_target:
            all_beaker_states, beaker_state_to_idx = gen_all_beaker_states("alchemy", args, encoding=encoding, tokenizer=tokenizer, device=args.device)
        # don't use agg here if you need to learn custom weights (i.e. using attn)
        # (otherwise have to re-encode each training step...)
        probe_agg_method = args.probe_agg_method if args.probe_agg_method == 'sum' or args.probe_agg_method == 'avg' else None
        if encoding == "NL":
            all_state_input_ids, all_state_attn_mask, all_state_vectors, all_state_index = encode_target_states(
                state_model, None, tokenizer, encode_init_state, probe_model, args, all_state_targets=all_beaker_states)
        else:
            # (numtotal,1,encodedim)
            all_state_input_ids = torch.stack(all_beaker_states, dim=0).unsqueeze(1)
            all_state_attn_mask = torch.ones(all_state_input_ids.size(0),1).to(args.device)
            all_beaker_states = all_state_input_ids.tolist()
        beaker_idx_to_state = {beaker_state_to_idx[state]: state for state in beaker_state_to_idx}
print("Initial eval + save model preds")
with torch.no_grad():
    probe_val_loss = 0
    n_val = 0
    n_init_state_match = 0
    n_exact_state_match = 0
    n_ratio_state_match = 0

    all_examples_output = {'prior': [], 'gold_state': [], 'gen_state': []}
    if not encode_tgt_state: all_examples_output = {**all_examples_output, 'prior': [], 'gold': [], 'gen': [], 'consistent': []}
    for j, (inputs, lang_tgts, probe_outs, raw_state_targets, init_states) in enumerate(tqdm(convert_to_transformer_batches(
        dev_dataset, tokenizer, EVAL_BATCHSIZE, include_init_state=encode_init_state, no_context=args.no_context,
        append_last_state_to_context=args.append_last_state_to_context, domain="alchemy", state_targets_type=probe_target,
        device=args.device, control_input=args.control_input,
    ))):
        if encode_tgt_state:
            '''
            encode_target_states implicitly created targets
            '''
            probe_outs['raw_inputs'] = raw_state_targets
            probe_outs['all_states_input_ids'] = all_state_input_ids
            if encoding == "NL":
                probe_outs['all_states_attn_mask'] = all_state_attn_mask
                probe_outs['all_states_encoding'] = all_state_vectors
            else:
                # (numtotal,1,embeddim)
                probe_outs['all_states_encoding'] = state_model(all_state_input_ids)
                probe_outs['all_states_attn_mask'] = all_state_attn_mask
            probe_outs['labels'] = get_matching_state_labels(all_beaker_states, beaker_state_to_idx, probe_outs, encode_tgt_state, tokenizer, device=args.device)
        model_outs = full_joint_model(inputs['input_ids'], inputs['attention_mask'], offset_mapping=inputs['offset_mapping'], probe_outs=probe_outs, localizer_key=inputs['state_key'])
        valid_loss = model_outs["loss"]
        probe_val_loss += valid_loss*len(inputs['input_ids'])
        n_val += len(inputs['input_ids'])
        if not encode_tgt_state:
            model_outs = ModelOutput(last_hidden_state=model_outs.last_hidden_state)
            generated_ws = full_joint_model.generate(
                input_ids=inputs['input_ids'], encoder_outputs=model_outs, max_length=128, decoder_start_token_id=probe_model.config.pad_token_id, no_repeat_ngram_size=0)
        else:
            similarity_scores = model_outs["similarity"]
            pred_matches = similarity_scores.argmax(-1)
            n_exact_state_match += int((pred_matches == probe_outs['labels']).sum())
            generated_ws = probe_outs['all_states_input_ids'][pred_matches]
        for k in range(len(inputs['input_ids'])):
            init_state = init_states[k]
            if not encode_tgt_state:
                genState = tokenizer.decode(generated_ws[k], skip_special_tokens=True)
            elif encode_tgt_state.split('.')[0] == 'NL':
                genState = beaker_idx_to_state[all_beaker_states['input_ids'].tolist().index(generated_ws[k].tolist())]
            else:
                genState = beaker_idx_to_state[all_beaker_states.index(generated_ws[k].tolist())]
            gtState = tokenizer.decode(probe_outs['input_ids'][k], skip_special_tokens=True)
            old_genstate, old_gtstate = genState, gtState
            if "the" in gtState and probe_target != "text":
                gtState = translate_nl_to_states(gtState, "alchemy")
                genState = translate_nl_to_states(genState, "alchemy")
            # otherwise already added
            if not encode_tgt_state: n_exact_state_match += int(genState == gtState)
            n_init_state_match += int(genState == init_state)
            n_ratio_state_match += similarity_fn(genState, gtState, "alchemy")
            all_examples_output['prior'].append(tokenizer.decode(inputs['input_ids'][k], skip_special_tokens=True))
            all_examples_output['gold_state'].append(gtState)
            all_examples_output['gen_state'].append(genState)

    print("n_val", n_val)
    if n_val == 0: n_val = 1
    avg_probe_val_loss = probe_val_loss/n_val
    decoder_metrics = (f", fraction {probe_target} match: {n_ratio_state_match/n_val}, " + \
        (f"percent init state match: {n_init_state_match/n_val}, " if probe_target == 'state' else '') + \
        f"percent exact match: {n_exact_state_match/n_val}")
    print(
        f"avg {probe_target} val loss {avg_probe_val_loss}{decoder_metrics}"
    )
# save states
if save_state and args.eval_only:
    new_all_examples_output = []
    for idx in range(len(all_examples_output['gold_state'])):
        new_all_examples_output.append({})
        for field in all_examples_output:
            if len(all_examples_output[field]) > 0:
                new_all_examples_output[idx][field] = all_examples_output[field][idx]
    with open(output_json_fn, "w") as wf:
        print(f"Saving to {output_json_fn}")
        for line in new_all_examples_output:
            wf.write(json.dumps(line) + "\n")
if args.eval_only:
    exit()


best_n_exact_state_match = n_exact_state_match
if not encode_tgt_state:
    best_n_similarity = n_ratio_state_match
else:
    best_val_loss = probe_val_loss
best_epoch = 0

init_state_dict = deepcopy(full_joint_model.state_dict())

# Training loops
print("Begin training")
for i in range(args.epochs):
    if (i - best_epoch > args.patience) and (i - best_epoch > args.patience): break
    model.train()
    probe_model.train()
    if encode_tgt_state: state_model.train()
    probe_train_losses = []

    for j, (inputs, lang_tgts, probe_outs, raw_state_targets, init_states) in enumerate(convert_to_transformer_batches(
        dataset, tokenizer, BATCHSIZE, random=random, include_init_state=encode_init_state, no_context=args.no_context,
        append_last_state_to_context=args.append_last_state_to_context, domain="alchemy", state_targets_type=probe_target,
        device=args.device, control_input=args.control_input,
    )):
        optimizer.zero_grad()
        if encode_tgt_state:
            probe_outs['raw_inputs'] = raw_state_targets
            probe_outs['all_states_input_ids'] = all_state_input_ids
            if encoding == "NL":
                probe_outs['all_states_attn_mask'] = all_state_attn_mask
                probe_outs['all_states_encoding'] = all_state_vectors
            else:
                # (numtotal,1,embeddim)
                probe_outs['all_states_encoding'] = state_model(all_state_input_ids)
                probe_outs['all_states_attn_mask'] = all_state_attn_mask
            probe_outs['labels'] = get_matching_state_labels(all_beaker_states, beaker_state_to_idx, probe_outs, encode_tgt_state, tokenizer, device=args.device)
        probe_loss = full_joint_model(inputs['input_ids'], inputs['attention_mask'], offset_mapping=inputs['offset_mapping'], probe_outs=probe_outs, localizer_key=inputs['state_key'])["loss"]
        probe_loss.backward()
        probe_train_losses.append(probe_loss)

        optimizer.step()
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, {probe_target} score {probe_loss.item()}", flush=True)
    print(f"epoch {i}, average {probe_target} loss {sum(probe_train_losses).item()/len(probe_train_losses) if len(probe_train_losses) > 0 else 0}")

    #dev loss
    model.eval()
    probe_model.eval()
    if encode_tgt_state: state_model.eval()
    with torch.no_grad():
        probe_val_loss = 0
        n_val = 0
        n_exact_state_match = 0
        n_ratio_state_match = 0
        for j, (inputs, lang_tgts, probe_outs, raw_state_targets, init_states) in enumerate(tqdm(convert_to_transformer_batches(
            dev_dataset, tokenizer, EVAL_BATCHSIZE, include_init_state=encode_init_state, no_context=args.no_context,
            append_last_state_to_context=args.append_last_state_to_context, domain="alchemy", state_targets_type=probe_target,
            device=args.device, control_input=args.control_input,
        ))):
            if encode_tgt_state:
                probe_outs['raw_inputs'] = raw_state_targets
                probe_outs['all_states_input_ids'] = all_state_input_ids
                if encoding == "NL":
                    probe_outs['all_states_attn_mask'] = all_state_attn_mask
                    probe_outs['all_states_encoding'] = all_state_vectors
                else:
                    # (numtotal,1,embeddim)
                    probe_outs['all_states_encoding'] = state_model(all_state_input_ids)
                    probe_outs['all_states_attn_mask'] = all_state_attn_mask
                probe_outs['labels'] = get_matching_state_labels(all_beaker_states, beaker_state_to_idx, probe_outs, encode_tgt_state, tokenizer, device=args.device)

            model_outs = full_joint_model(inputs['input_ids'], inputs['attention_mask'], offset_mapping=inputs['offset_mapping'], probe_outs=probe_outs, localizer_key=inputs['state_key'])
            valid_loss = model_outs["loss"]
            probe_val_loss += valid_loss*len(inputs['input_ids'])
            n_val += len(inputs['input_ids'])
            if not encode_tgt_state:
                model_outs = ModelOutput(last_hidden_state=model_outs.last_hidden_state)
                generated_ws = full_joint_model.generate(
                    input_ids=inputs['input_ids'], encoder_outputs=model_outs, max_length=128, decoder_start_token_id=probe_model.config.pad_token_id, no_repeat_ngram_size=0)
                for k in range(len(generated_ws)):
                    genState = tokenizer.decode(generated_ws[k], skip_special_tokens=True)
                    gtState = tokenizer.decode(probe_outs['input_ids'][k], skip_special_tokens=True)
                    n_exact_state_match += int(genState == gtState)
                    n_ratio_state_match += similarity_fn(genState, gtState, "alchemy")
            else:
                similarity_scores = model_outs["similarity"]
                n_exact_state_match += int((similarity_scores.argmax(-1) == probe_outs['labels']).sum())
        print("n_val", n_val)
        avg_probe_val_loss = probe_val_loss.item()/n_val
        if not encode_tgt_state:
            decoder_metrics = f", fraction {probe_target} match: {n_ratio_state_match/n_val}, percent exact match: {n_exact_state_match/n_val}"
        else: decoder_metrics = f", accuracy: {n_exact_state_match/n_val}"
        print(
            f"epoch {i}, avg {probe_target} val loss {avg_probe_val_loss}{decoder_metrics}"
        )
        if best_n_exact_state_match < n_exact_state_match or (best_n_exact_state_match == n_exact_state_match and (
            (not encode_tgt_state and best_n_similarity < n_ratio_state_match) or (encode_tgt_state and probe_val_loss < best_val_loss)
        )):
            print(f"NEW BEST MODEL: {best_n_exact_state_match/n_val} -> {n_exact_state_match/n_val}")
            model.epoch = i
            best_n_exact_state_match = n_exact_state_match
            best_n_similarity = n_ratio_state_match
            best_val_loss = probe_val_loss
            torch.save(probe_model.state_dict(), probe_save_path)
            best_epoch = i

