import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np

import textworld
from transformers import BartConfig, T5Config
from transformers import BartTokenizerFast, T5TokenizerFast
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import AdamW

import argparse
import os
import glob
from tqdm import tqdm
import copy
import json
import logging
import random

from localizer.tw_localizer import TWLocalizer
from metrics.tw_metrics import get_em, get_confusion_matrix
from data.textworld.parse_tw import (
    translate_inv_items_to_str, translate_inv_str_to_items,
)
from data.textworld.utils import (
    EntitySet, control_mention_to_tgt_simple, control_mention_to_tgt_with_rooms_simple,
    load_possible_pairs, load_negative_tgts,
)
from data.textworld.tw_dataloader import (
    TWDataset, TWEntitySetDataset, TWFullDataLoader, TWEntitySetDataLoader,
)

from probe_models import (
    ProbeLinearModel, ProbeConditionalGenerationModel, ProbeLanguageEncoder, encode_target_states,
    get_probe_model, get_state_encoder, get_lang_model,
)
from itertools import chain, combinations

DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='bart', choices=['bart', 't5'])
parser.add_argument('--override_num_layers', type=int, default=None)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--eval_batchsize', type=int, default=128)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_only', default=False, action='store_true')
parser.add_argument('--gamefile', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--no_pretrain', default=False, action='store_true')
parser.add_argument('--control_input', default=False, action='store_true', help='control inputs to tokenized entity pair')
parser.add_argument('--local_files_only', action='store_true', default=False)
parser.add_argument('--lm_save_path', type=str, default=None)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--metric', type=str, choices=['em', 'loss'], help='which metric to use on dev set', default='em')
parser.add_argument('--probe_save_path', type=str, default=None)
parser.add_argument('--probe_layer', type=int, default=-1, help="which layer of the model to probe")
parser.add_argument('--probe_type', type=str, choices=['3linear_classify', 'linear_classify', 'linear_retrieve', 'decoder'], default='decoder')
parser.add_argument('--encode_tgt_state', type=str, default=False, choices=[False, 'NL.bart', 'NL.t5'], help="how to encode the state before probing")
parser.add_argument('--train_data_size', type=int, default=4000)
parser.add_argument('--tgt_agg_method', type=str, choices=['sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default='avg', help="how to aggregate across tokens of target, if `encode_tgt_state` is set True")
parser.add_argument('--probe_agg_method', type=str, choices=[None, 'sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default=None, help="how to aggregate across tokens")
parser.add_argument('--probe_attn_dim', type=int, default=None, help="what dimensions to compress sequence tokens to")
parser.add_argument('--probe_target', type=str, default='final.belief_facts', choices=list(chain(*[[
    f'{init_final}.full_facts', f'{init_final}.full_belief_facts', f'{init_final}.belief_facts',
    f'{init_final}.belief_facts_single', f'{init_final}.belief_facts_pair',
    f'{init_final}.full_belief_facts_single', f'{init_final}.full_belief_facts_pair',
    f'{init_final}.belief_facts_single.control', f'{init_final}.full_belief_facts_single.control', f'{init_final}.belief_facts_single.control_with_rooms', f'{init_final}.full_belief_facts_single.control_with_rooms',
    f'{init_final}.belief_facts_pair.control', f'{init_final}.full_belief_facts_pair.control', f'{init_final}.belief_facts_pair.control_with_rooms', f'{init_final}.full_belief_facts_pair.control_with_rooms',
] for init_final in ['init', 'final']])))
parser.add_argument('--localizer_type', type=str, default='all',
    choices=['all'] + [f'belief_facts_{sides}_{agg}' for sides in ['single', 'pair'] for agg in ['all', 'first', 'last']],
    help="which encoded tokens of the input to probe."
    "Set to `all`, `belief_facts_{single|pair}_{all|first|last}`")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ents_to_states_file', type=str, default=None, help='Filepath to precomputed state vectors')
args = parser.parse_args()

arch = args.arch
pretrained = not args.no_pretrain
batchsize = args.batchsize
control_input = args.control_input
eval_batchsize = args.eval_batchsize
lm_save_path = args.lm_save_path
localizer_type = args.localizer_type
probe_target = args.probe_target.split('.')
probe_type = args.probe_type
retrieve = probe_type.endswith('retrieve')
classify = probe_type.endswith('classify')
assert not (retrieve and classify)
encode_tgt_state = args.encode_tgt_state
tgt_agg_method = args.tgt_agg_method
probe_agg_method = args.probe_agg_method
probe_attn_dim = args.probe_attn_dim
probe_save_path = args.probe_save_path
train_data_size = args.train_data_size
game_kb = None
inform7_game = None
# NOTE: inexact if grammars between worlds are different (not the case, at least for simple domain)
for fn in glob.glob(os.path.join(args.gamefile, 'train/*.ulx')):
    env = textworld.start(fn)
    game_state = env.reset()
    game_kb = game_state['game'].kb.inform7_predicates
    inform7_game = env._inform7
    break
max_seq_len = args.max_seq_len
metric = args.metric
control_mention_to_tgt, control_mention_to_tgt_with_rooms = control_mention_to_tgt_simple, control_mention_to_tgt_with_rooms_simple

# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)


def eval_model(args, model, dev_dataloader, tokenizer, eval_batchsize, precomputed_negs=None, output_json_fn=None):
    model.eval()
    with torch.no_grad():
        prev_prev_context = None
        tot_val_loss = 0
        n_val = 0
        em = 0
        f1 = 0
        cms = None
        saved_preds = []
        save_dict = []
        for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entity_sets) in enumerate(tqdm(dev_dataloader)):
            if inputs is None: continue
            bs = len(game_ids)
            model_outs = model(inputs['input_ids'], inputs['attention_mask'], offset_mapping=inputs['offset_mapping'], probe_outs=tgt_state, localizer_key=entity_sets['mentions'])
            tot_val_loss += model_outs["loss"] * len(inputs['input_ids'])
            n_val += len(inputs['input_ids'])
            gold_state = []
            if not encode_tgt_state:
                # (1024 if args.eval_only else 128)
                generated_state = model.generate(
                    encoder_outputs=model_outs["decoder_inputs"], max_length=128, decoder_start_token_id=model.probe_base_model.config.pad_token_id, do_sample=False,
                    no_repeat_ngram_size=0)
            else:
                out_texts = {}
                similarity_scores = model_outs["similarity"]
                if retrieve:
                    pred_matches = similarity_scores.argmax(-1)
                    em += int((pred_matches == tgt_state['labels']).sum())
                    generated_state = tgt_state['all_states_input_ids'][pred_matches]
                elif classify:
                    # (bs, # ents/facts)
                    label_mask = tgt_state['labels'] != -1
                    pred_matches = similarity_scores.argmax(-1)
                    pred_matches[~label_mask] = -1
                    em += int(((pred_matches != tgt_state['labels']).sum(1) == 0).sum())
                    save_dict.append({'score': similarity_scores, 'labels': tgt_state['labels']})
                    cm = get_confusion_matrix(pred_matches, tgt_state['labels'], 3)
                    if cms is None: cms = cm
                    else: cms = [[cm[i][j] + cms[i][j] for j in range(len(cms[i]))] for i in range(len(cms))]
                    # (bs,)
                    generated_state = []
                    for i, item in enumerate(pred_matches):
                        entity_list = entity_sets['entities'][i]
                        tgt_state['all_states_input_ids'] = tgt_state['all_states_input_ids'].view(bs, -1, tgt_state['all_states_input_ids'].size(-1))
                        idx_to_state = [tokenizer.decode(neg_inp_ids, skip_special_tokens=True) for neg_inp_ids in tgt_state['all_states_input_ids'][i]]
                        generated_state.append({
                            "true": [idx_to_state[idx] for idx in (item == 1).nonzero(as_tuple=False)[:,0]],
                            "false": [idx_to_state[idx] for idx in (item == 2).nonzero(as_tuple=False)[:,0]],
                        })
                        gold_state.append({
                            "true": [idx_to_state[idx] for idx in (tgt_state['labels'][i] == 1).nonzero(as_tuple=False)[:,0]],
                            "false": [idx_to_state[idx] for idx in (tgt_state['labels'][i] == 2).nonzero(as_tuple=False)[:,0]],
                        })
            for i, gi in enumerate(generated_state):
                if len(gold_state) > i: gt_state = gold_state[i]
                gen_state = generated_state[i]
                if not encode_tgt_state: # otherwise already added
                    gt_state = tokenizer.decode(tgt_state['input_ids'][i], skip_special_tokens=True)
                    gen_state = tokenizer.decode(gen_state, skip_special_tokens=True)
                    matches = get_em(gt_state, gen_state, probe_target[1])
                    em += matches[0]
                    f1 += matches[1]
                    gt_state, gen_state = list(matches[2]), list(matches[3])
                if output_json_fn:
                    prev_context = dev_dataset[j*eval_batchsize+i]['contexts']
                    if not args.control_input:
                        assert tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True).strip() == prev_context.strip()
                    saved_preds.append({
                        'prev_context': prev_context, 'gt_state': gt_state, 'gen_state': gen_state, 'game_id': game_ids[i],
                    })
                    if prev_prev_context is None or prev_context != prev_prev_context:
                        prev_prev_context = prev_context
                    if entity_sets:
                        saved_preds[-1]['entity'] = EntitySet.serialize(entity_sets['entities'][i])
    if output_json_fn is not None:
        os.makedirs(os.path.split(output_json_fn)[0], exist_ok=True)
        with open(output_json_fn, 'w') as wf:
            for pred in saved_preds:
                wf.write(json.dumps(pred) + '\n')
        torch.save(save_dict, output_json_fn.replace('.jsonl', '.pt_preds'))
        print(f"Saved model prediction to {output_json_fn}")

    print("n_val", n_val)
    avg_val_loss = tot_val_loss / n_val
    avg_em = em / n_val
    if cms is None:
        avg_f1 = f1 / n_val
        return n_val, avg_val_loss, avg_em, avg_f1
    else:
        # (pred, label, n_val)
        cms = torch.tensor(cms)
        avg_cm = cms.sum(-1).float() / n_val
        return n_val, avg_val_loss, avg_em, avg_cm

# create save dir/path
save_state = True 
if lm_save_path:
    save_dir = os.path.basename(lm_save_path.split('.')[0])
elif pretrained:
    save_dir = f"noft_{arch}"
else:
    save_dir = f"nopre_noft_{arch}"
output_dir = os.path.join(save_dir, f"{'enctgt_' if encode_tgt_state else ''}{probe_type}" + \
    f"_{localizer_type + (f'.control_inp' if control_input else '')}_{probe_agg_method}{probe_attn_dim if probe_agg_method and probe_agg_method.endswith('_attn') else ''}{tgt_agg_method if encode_tgt_state else ''}" + \
    f"_{args.probe_target}{'_'+str(train_data_size)}_seed{args.seed}")
if not probe_save_path:
    probe_save_path = os.path.join("probe_models_textworld", f"{output_dir}.p")
os.makedirs(os.path.split(probe_save_path)[0], exist_ok=True)

# load (language) model
model, encoder, tokenizer = get_lang_model(arch, lm_save_path, pretrained, local_files_only=args.local_files_only, n_layers=args.override_num_layers, device=args.device)

# create/load world state encoder
# (w/ same encoder as LM)
state_model = None
if encode_tgt_state:
    state_model = get_state_encoder(
        encode_tgt_state.split('.')[-1], encoder, config=model.config, pretrained=pretrained,
        freeze_params=True, local_files_only=args.local_files_only, n_layers=args.override_num_layers, device=args.device,
    )

# create probe model
probe_model = get_probe_model(probe_type, probe_agg_method, probe_attn_dim, arch, model, probe_save_path, args.tgt_agg_method, encode_tgt_state=args.encode_tgt_state, local_files_only=args.local_files_only, device=args.device)

localizer = TWLocalizer(getattr(probe_model, 'agg_layer', None), args.probe_agg_method, args.probe_attn_dim, args.localizer_type, None, tokenizer, args.device)
state_localizer = TWLocalizer(getattr(probe_model, 'target_agg_layer', None), args.tgt_agg_method, args.probe_attn_dim, 'all', None, tokenizer, args.device)
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
optimizer = AdamW([p for p in full_joint_model.parameters() if p.requires_grad], lr=args.lr)
print("Loaded model")

print(f"Saving probe checkpoints to {probe_save_path}")
output_json_fn = None
if args.eval_only:
    output_json_dir = os.path.split(probe_save_path)[0]
    if not os.path.exists(output_json_dir): os.makedirs(output_json_dir)
    output_json_fn = os.path.join(output_json_dir, f"{os.path.split(probe_save_path)[-1].replace('.p', '.jsonl')}")
    print(f"Saving predictions to {output_json_fn}")

if DEBUG:
    max_data_size = [2,10]
else:
    max_data_size = [500,train_data_size]

state_key = probe_target[1].replace('_single', '').replace('_pair', '')
tgt_state_key = probe_target[0]+'_states'
possible_pairs = None
if probe_target[1].endswith('_pair'):
    ent_set_size = 2
    possible_pairs = load_possible_pairs(pair_out_file=os.path.join(args.data, 'entity_pairs.json'))
    assert possible_pairs is not None
if probe_target[1].endswith('_single'): ent_set_size = 1
neg_facts_fn = args.ents_to_states_file
if neg_facts_fn is None:
    neg_facts_fn = os.path.join(
        os.path.join(args.data, 'entities_to_facts'),
        f'{probe_target[1]}_{arch}_state_model_{os.path.split(lm_save_path)[-1].replace(".p", "")}.p'
    )
precomputed_negs = load_negative_tgts(negative_tgts_fn=neg_facts_fn)
assert precomputed_negs is not None

control = probe_target[2] if len(probe_target)>2 else False
# load data
dev_dataset = TWEntitySetDataset(
    args.data, tokenizer, 'dev', max_seq_len=max_seq_len, ent_set_size=ent_set_size, control=control,
    gamefile=args.gamefile, state_key=state_key, tgt_state_key=tgt_state_key, max_data_size=max_data_size[0],
    inform7_game=inform7_game, possible_pairs=possible_pairs, precomputed_negs=precomputed_negs,
)
dataset = TWEntitySetDataset(
    args.data, tokenizer, 'train', max_seq_len=max_seq_len, ent_set_size=ent_set_size, control=control,
    gamefile=args.gamefile, state_key=state_key, tgt_state_key=tgt_state_key, max_data_size=max_data_size[1],
    inform7_game=inform7_game, possible_pairs=possible_pairs, precomputed_negs=precomputed_negs,
)
print(f"Loaded data: {len(dataset)} train examples, {len(dev_dataset)} dev examples")
train_dataloader = TWEntitySetDataLoader(dataset, tokenizer, batchsize, control_input, device=args.device)
dev_dataloader = TWEntitySetDataLoader(dev_dataset, tokenizer, eval_batchsize, control_input, device=args.device)
print("Created batches")

# get all pairs of entities to query
game_ids = dev_dataset.get_gameids() + dataset.get_gameids()
# initial eval
print("Initial eval")
avg_val_loss, avg_em, avg_overlap = 0,0,0
n_val, avg_val_loss, avg_em, avg_overlap = eval_model(args, full_joint_model, dev_dataloader, tokenizer, eval_batchsize, precomputed_negs=precomputed_negs, output_json_fn=output_json_fn)
print(f"INIT, avg val loss: {avg_val_loss}, avg em: {avg_em}, avg overlap: {avg_overlap}")
best_val_loss = avg_val_loss
best_em = avg_em
best_epoch = -1

if args.eval_only:
    exit()

# training loop
print("Start training")
for i in range(args.epochs):
    if i - best_epoch > 10: break
    full_joint_model.train()
    train_losses = []
    for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entity_sets) in enumerate(train_dataloader):
        if j % 1000 == 0 and j != 0:
            # do eval
            n_val, avg_val_loss, avg_em, avg_overlap = eval_model(args, full_joint_model, dev_dataloader, tokenizer, eval_batchsize, precomputed_negs=precomputed_negs)
            print(f"epoch {i} update {j}, avg val loss: {avg_val_loss}, avg em: {avg_em}, avg overlap: {avg_overlap}")
            if (metric == 'em' and avg_em >= best_em) or (metric == 'loss' and avg_val_loss <= best_val_loss):
                print("NEW BEST MODEL")
                probe_model.epoch = i
                best_val_loss = avg_val_loss
                best_em = avg_em
                torch.save(probe_model.state_dict(), probe_save_path)
                best_epoch = i
            else:
                print(f"model {metric} went {'down' if metric == 'em' else 'up'}")

        if inputs is None: continue
        optimizer.zero_grad()
        model_outs = full_joint_model(inputs['input_ids'], inputs['attention_mask'], offset_mapping=inputs['offset_mapping'], probe_outs=tgt_state, localizer_key=entity_sets['mentions'])
        probe_loss = model_outs["loss"]
        train_losses.append(probe_loss.item())
        probe_loss.backward()
        optimizer.step()
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, loss: {probe_loss.item()}", flush=True)

    # do eval
    n_val, avg_val_loss, avg_em, avg_overlap = eval_model(args, full_joint_model, dev_dataloader, tokenizer, eval_batchsize, precomputed_negs=precomputed_negs)
    print(f"epoch {i}, avg val loss: {avg_val_loss}, avg em: {avg_em}, avg overlap: {avg_overlap}")
    if (metric == 'em' and avg_em >= best_em) or (metric == 'loss' and avg_val_loss <= best_val_loss):
        print("NEW BEST MODEL")
        probe_model.epoch = i
        best_val_loss = avg_val_loss
        best_em = avg_em
        torch.save(probe_model.state_dict(), probe_save_path)
        best_epoch = i
    else:
        print(f"model {metric} went {'down' if metric == 'em' else 'up'}")
