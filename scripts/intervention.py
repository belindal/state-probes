from typing import Iterable, List, Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput
from torch import nn
import numpy as np
import random

from parseScone import loadData, getBatches, getBatchesWithInit
import os
import json
from tqdm import tqdm

from data.alchemy.alchemy_artificial_generator import execute
from data.alchemy.parse_alchemy import consistencyCheck, parse_world, parse_utt_with_world, word_to_int, colors
from utils import (
    check_val_consistency, batch, DEVICE, int_to_word,
    ContrastiveClassifierHead, DumbStateEncoder, TransformerStateEncoder,
)
from data.alchemy.scone_dataloader import convert_to_transformer_batches
from probe_models import (
    get_lang_model,
)
import Levenshtein
import torch.nn.functional as F
from utils import translate_nl_to_states, translate_states_to_nl

import faiss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eval_batchsize', type=int, default=128)
parser.add_argument('--arch', type=str, default='bart', choices=['t5', 'bart'])
parser.add_argument('--encode_init_state', type=str, default='NL', choices=[False, 'raw', 'NL'])
parser.add_argument('--seed', type=int, default=45)
parser.add_argument('--lm_save_path', type=str, default=None, help="load existing LM checkpoint (if any)")
parser.add_argument('--probe_layer', type=int, default=-1, help="which layer of the model to probe")
parser.add_argument('--overwrite_save', action='store_true', default=False)
parser.add_argument('--create_type', type=str, choices=['drain_1', 'pour_1', 'drain_half'], help='what command(s) to append')
args = parser.parse_args()

EVAL_BATCHSIZE = args.eval_batchsize
arch = args.arch
probe_layer = args.probe_layer
encode_init_state = args.encode_init_state
lm_save_path = args.lm_save_path
create_type = args.create_type
fn = f"replace_states_outs/{args.lm_save_path.split('/')[-1].split('.')[0]}_append_both_{create_type}.jsonl"
update_orig = True


# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# must replace the beaker tokens at the appropriate location

replaced_inp_sentence = None

# load (language) model
base_lm, encoder, tokenizer = get_lang_model(arch, lm_save_path)

def encode_input(inp_sentence=None, tokenized_sentence=None):
    if tokenized_sentence is None:
        inputs = tokenizer(inp_sentence, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    else:
        inputs = tokenized_sentence
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    # (bs, seqlen, hiddim); [(bs, seqlen, hiddim) x layers]
    encoder_outputs = base_lm.get_encoder()(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, output_attentions=False,
    )
    # (bsz, seqlen, embed_dim) = [(bsz, seqlen, embed_dim) x num_layers] [probe_layer]
    encoder_outputs.last_hidden_state = encoder_outputs.hidden_states[args.probe_layer]
    encoder_outputs.hidden_states = None
    return encoder_outputs, input_ids, attention_mask


def replace_tokens(
    hidden_reps, input_ids, attention_mask,
    replace_hidden_reps, replace_input_ids, replace_attention_mask,
    beaker_to_replace,
):
    '''
    One instance from a batch
    hidden_reps: seqlen x embed_dim
    '''
    hidden_reps_mask = attention_mask
    replace_hidden_reps_mask = replace_attention_mask

    if replace_input_ids is not None:
        for btr in beaker_to_replace:
            # int_to_word[beaker_to_replace]
            split_pos = (input_ids == tokenizer.convert_tokens_to_ids(",")).nonzero(as_tuple=False).squeeze(-1)
            period_pos = (input_ids == tokenizer.convert_tokens_to_ids(".")).nonzero(as_tuple=False).squeeze(-1)[0]
            if btr == 0: begin_pos = 0
            else: begin_pos = split_pos[btr - 1]
            if btr == 6: end_pos = period_pos
            else: end_pos = split_pos[btr]

            replace_split_pos = (replace_input_ids == tokenizer.convert_tokens_to_ids(",")).nonzero(as_tuple=False).squeeze(-1)
            replace_period_pos = (replace_input_ids == tokenizer.convert_tokens_to_ids(".")).nonzero(as_tuple=False).squeeze(-1)[0]
            if btr == 0: replace_begin_pos = 0
            else: replace_begin_pos = replace_split_pos[btr - 1]
            if btr == 6: replace_end_pos = period_pos
            else: replace_end_pos = replace_split_pos[btr]

            assert (replace_input_ids[begin_pos:end_pos][:5] == input_ids[begin_pos:end_pos][:5]).all()
            input_ids = torch.cat([input_ids[:begin_pos], replace_input_ids[replace_begin_pos:replace_end_pos], input_ids[end_pos:]], dim=0)
            hidden_reps = torch.cat([hidden_reps[:begin_pos,:], replace_hidden_reps[replace_begin_pos:replace_end_pos,:], hidden_reps[end_pos:,:]], dim=0)
            hidden_reps_mask = torch.cat([hidden_reps_mask[:begin_pos], replace_hidden_reps_mask[replace_begin_pos:replace_end_pos], hidden_reps_mask[end_pos:]], dim=0)
    return hidden_reps, hidden_reps_mask


def get_log_likelihood_of_samples(base_lm, encoder_outputs, samples):
    if encoder_outputs.size(0) == 1 and samples.size(0) != 1: encoder_outputs = torch.cat(samples.size(0) * [encoder_outputs])
    base_lm_token_logits = base_lm(input_ids=None, encoder_outputs=(encoder_outputs, None), labels=samples)[1]
    base_lm_token_logprobs = base_lm_token_logits.log_softmax(-1)
    target_token_logprobs = base_lm_token_logprobs.gather(dim=-1, index=samples.unsqueeze(-1)).squeeze(-1)
    target_sentence_logprobs = target_token_logprobs.sum(-1)
    return target_sentence_logprobs


def make_list_tensor(lst, mask, pad_idx, pad_dim):
    # make list into tensor (w/ padding)
    # lst is list of tensors
    max_len = max([t.size(pad_dim) for t in lst])
    tensor = [F.pad(t, pad=tuple([0 for _ in range(2*(len(t.size())-1-pad_dim)+1)] + [max_len-t.size(pad_dim)]), value=pad_idx) for t in lst]
    tensor = torch.cat(tensor, dim=0)
    mask_tensor = [F.pad(t, pad=tuple([0 for _ in range(2*(len(t.size())-1-pad_dim)+1)] + [max_len-t.size(pad_dim)]), value=0) for t in mask]
    mask_tensor = torch.cat(mask_tensor, dim=0)
    return tensor, mask_tensor


def gen_all_actions(action_type):
    # drain {1|2|3|4} from the {first |second |...|last | }{|orange|red|...} beaker
    # pour the {first|second|...|last| } {|orange|red|...} beaker to the {first|second|...|last| } {|orange|red|...} beaker
    # mix beaker {0|1|2|...|7}
    all_positions = [f'{pos} ' for pos in list(word_to_int.keys()) + ['']]
    all_beakers = [f'{pos}{color}' for pos in all_positions for color in colors] + list(word_to_int.keys())
    all_actions = []
    if action_type == 'all' or action_type == 'drain':
        all_actions += [f'drain {amt} from the {beaker} beaker' for amt in range(1,5) for beaker in all_beakers]
    if action_type == 'all' or action_type == 'pour':
        all_actions += [f'pour the {beaker1} beaker to the {beaker2} beaker' for beaker1 in all_beakers for beaker2 in all_beakers]
    if action_type == 'all' or action_type == 'mix':
        all_actions += [f'mix beaker {bp_idx}' for bp_idx in range(7)]
    return all_actions


def create_new_sentence(next_utt, priorTxt, priorTxt_rawstate, orig_final_state_raw, create_type):
    """
    Let target_beaker=beaker referred to in `next_utt`
    drain_1:
    new_sentence = priorTxt + 'drain [amt in target_beaker] from [target_beaker]'
    """
    try:
        assert consistencyCheck(priorTxt_rawstate, next_utt)
        # find beaker to target (i.e. beaker in the utterance, which we empty from)
        action, action_args = parse_utt_with_world(next_utt[:-1], orig_final_state_raw)
        target_beaker_pos = action_args[0]  # draining from, pouring from, or mixing this beaker (0-indexed)
    except (KeyError, AssertionError) as e:
        # random nonempty beaker
        for target_beaker_pos in range(7):
            if orig_final_state_raw['objects'][target_beaker_pos] is not None and len(orig_final_state_raw['objects'][target_beaker_pos]) > 0: break
    assert orig_final_state_raw['objects'][target_beaker_pos] is not None  # since we'll be doing an action on it...
    target_beaker_amt = len(orig_final_state_raw['objects'][target_beaker_pos])
    if create_type == "drain_1":
        new_next_utt = f"drain {target_beaker_amt} from the {int_to_word[target_beaker_pos]} beaker"
        target_beaker_pos = [target_beaker_pos]
    elif create_type == "drain_half":
        all_beaker_amts = [len(beaker) if beaker is not None else 0 for beaker in orig_final_state_raw['objects']]
        pos_to_include = [pos for pos in range(len(all_beaker_amts)) if all_beaker_amts[pos] > 0]
        # take approx half
        pos_to_include = pos_to_include[:int(len(pos_to_include)/2)]
        new_next_utt = ".\n".join([f"drain {all_beaker_amts[pos]} from the {int_to_word[pos]} beaker" for pos in pos_to_include])
        target_beaker_pos = pos_to_include
    elif create_type == "pour_1":
        all_beaker_amts = [len(beaker) if beaker is not None else 0 for beaker in orig_final_state_raw['objects']]
        beaker_from = target_beaker_pos
        # find beaker w/ smallest amount (excluding empty) and target beaker
        beaker_amts_list = [
            len(orig_final_state_raw['objects'][beaker])
            if orig_final_state_raw['objects'][beaker] is not None and beaker != beaker_from else 1000
            for beaker in range(7)]
        beaker_to_amt, beaker_to = np.array(beaker_amts_list).min(), np.array(beaker_amts_list).argmin()
        assert beaker_to_amt <= 4 and beaker_to != beaker_from and orig_final_state_raw['objects'][beaker_to] is not None  # lowest beaker is not full
        if target_beaker_amt + beaker_to_amt > 4:
            amt_to_drain = beaker_to_amt + target_beaker_amt - 4
            beaker_to_drain = beaker_to if amt_to_drain <= beaker_to_amt else beaker_from
            # print(f"so rare case of additional draining {amt_to_drain} from target happening...")
            new_next_utt_init = f"drain {amt_to_drain} from the {int_to_word[beaker_to_drain]} beaker.\n"
        else: new_next_utt_init = ""
        # create new sentence by adding `pour`
        new_next_utt = f"{new_next_utt_init}pour the {int_to_word[beaker_from]} beaker to the {int_to_word[beaker_to]} beaker"
        target_beaker_pos = [beaker_from, beaker_to]
    else: raise NotImplementedError()

    new_sentence, new_sentence_rawstate, new_final_state_raw = priorTxt, priorTxt_rawstate, orig_final_state_raw
    for nnu in new_next_utt.split(".\n"):
        assert consistencyCheck(new_sentence_rawstate, nnu)
        # create new sentence by adding `drain` or `pour`
        new_sentence = f"{new_sentence}.\n{nnu}"
        new_sentence_rawstate = f"{new_sentence_rawstate}.\n{nnu}"
        # update world according to new sentence
        new_action, new_args = parse_utt_with_world(nnu, new_final_state_raw)
        new_final_state_raw = execute(new_final_state_raw, new_action, new_args)
    return new_sentence, new_sentence_rawstate, new_final_state_raw, target_beaker_pos


def create_orig_sentence(new_next_utt, new_sentence, new_sentence_rawstate, new_final_state_raw, priorTxt, priorTxt_rawstate, orig_final_state_raw, create_type):
    """
    Creating the new `original sentence` after creating `new sentence`
    Let target_beaker=beaker referred to in `new_next_utt`
    drain_1:
    orig_sentence = priorTxt + 'drain [amt in target_beaker] from [target_beaker]'
    """
    if consistencyCheck(new_sentence_rawstate, new_next_utt):
        # find beaker to target (i.e. beaker in the next utterance from `new`, which we empty from)
        action, action_args = parse_utt_with_world(new_next_utt[:-1], new_final_state_raw)
        orig_target_beaker_pos = action_args[0]  # draining from, pouring from, or mixing this beaker (0-indexed)
    else:
        # random nonempty beaker
        for orig_target_beaker_pos in range(7):
            if new_final_state_raw['objects'][orig_target_beaker_pos] is not None and len(new_final_state_raw['objects'][orig_target_beaker_pos]) > 0: break
    target_beaker_amt = len(orig_final_state_raw['objects'][orig_target_beaker_pos])
    if create_type == "drain_1":
        assert target_beaker_amt == len(new_final_state_raw['objects'][orig_target_beaker_pos])
        orig_next_utt = f"drain {target_beaker_amt} from the {int_to_word[orig_target_beaker_pos]} beaker"
    elif create_type == "drain_half":
        new_sentence_utt = new_sentence.replace(priorTxt + ".\n", "")
        new_all_beaker_amts = [len(beaker) if beaker is not None else 0 for beaker in new_final_state_raw['objects']]
        pos_to_include = [pos for pos in range(len(new_all_beaker_amts)) if new_all_beaker_amts[pos] > 0]
        orig_next_utt = ".\n".join([f"drain {new_all_beaker_amts[pos]} from the {int_to_word[pos]} beaker" for pos in pos_to_include])
        target_beaker_pos = pos_to_include
    elif create_type == "pour_1":
        all_beaker_amts = [len(beaker) if beaker is not None else 0 for beaker in orig_final_state_raw['objects']]
        new_sentence_utt = new_sentence.replace(priorTxt + ".\n", "")
        init_utt = "" if len(new_sentence_utt.split('.\n')) == 1 else new_sentence_utt.split('.\n')[0]
        new_sentence_utt = new_sentence_utt.split('.\n')[-1]
        if len(init_utt) == 0: init_utt = ""
        else: init_utt += ".\n"
        new_sentence_utt = new_sentence_utt.split("pour the ")[1][:-len(" beaker")]
        beaker_from, beaker_to = new_sentence_utt.split(" beaker to the ")[0], new_sentence_utt.split(" beaker to the ")[1]
        beaker_from, beaker_to = word_to_int[beaker_from], word_to_int[beaker_to]
        # apparently not always mix...
        # swap the beakers from the utterance in `new_sentence`
        orig_next_utt = f"{init_utt}pour the {int_to_word[beaker_to]} beaker to the {int_to_word[beaker_from]} beaker"
    else: raise NotImplementedError()

    orig_sentence, orig_sentence_rawstate = priorTxt, priorTxt_rawstate
    for onu in orig_next_utt.split(".\n"):
        assert consistencyCheck(orig_sentence_rawstate, onu)
        # create next sentence by adding `drain` or `pour`
        orig_sentence = f"{orig_sentence}.\n{onu}"
        orig_sentence_rawstate = f"{orig_sentence_rawstate}.\n{onu}"
        # update world according to next sentence
        next_action, next_args = parse_utt_with_world(onu, orig_final_state_raw)
        orig_final_state_raw = execute(orig_final_state_raw, next_action, next_args)
    return orig_sentence, orig_sentence_rawstate, orig_final_state_raw


def get_kl_table(orig_sentence_log_likelihood, mix_sentence_log_likelihood, new_sentence_log_likelihood):
    kl_divergences = [[0 for _ in range(3)] for _ in range(3)]
    kl_divergences[0][0] = (orig_sentence_log_likelihood.exp() * (orig_sentence_log_likelihood - orig_sentence_log_likelihood)).sum().item()
    kl_divergences[0][1] = (orig_sentence_log_likelihood.exp() * (orig_sentence_log_likelihood - mix_sentence_log_likelihood)).sum().item()
    kl_divergences[0][2] = (orig_sentence_log_likelihood.exp() * (orig_sentence_log_likelihood - new_sentence_log_likelihood)).sum().item()
    kl_divergences[1][0] = (mix_sentence_log_likelihood.exp() * (mix_sentence_log_likelihood - orig_sentence_log_likelihood)).sum().item()
    kl_divergences[1][1] = (mix_sentence_log_likelihood.exp() * (mix_sentence_log_likelihood - mix_sentence_log_likelihood)).sum().item()
    kl_divergences[1][2] = (mix_sentence_log_likelihood.exp() * (mix_sentence_log_likelihood - new_sentence_log_likelihood)).sum().item()
    kl_divergences[2][0] = (new_sentence_log_likelihood.exp() * (new_sentence_log_likelihood - orig_sentence_log_likelihood)).sum().item()
    kl_divergences[2][1] = (new_sentence_log_likelihood.exp() * (new_sentence_log_likelihood - mix_sentence_log_likelihood)).sum().item()
    kl_divergences[2][2] = (new_sentence_log_likelihood.exp() * (new_sentence_log_likelihood - new_sentence_log_likelihood)).sum().item()
    return kl_divergences


# load data
print("Loading data")
dataset, lang_v, state_v = loadData(split="train", kind="alchemy", synthetic=True)
dev_dataset, lang_v_dev, state_v_dev = loadData(split="dev", kind="alchemy", synthetic=True)
best_val_loss = 10**10
best_epoch = -1
all_train_states = [" ".join(state) for _, _, state in [x for d in dataset for x in d.all_pairs()]]
all_dev_states = [" ".join(state) for _, _, state in [x for d in dev_dataset for x in d.all_pairs()]]
random.shuffle(all_dev_states)
#dev loss
base_lm.eval()

kl_divergences_all = []
print(fn)
if args.overwrite_save or not os.path.exists(fn):
    existing_result_ids = {}
else:
    existing_lines = open(fn).readlines()
    existing_lines = [json.loads(line) for line in existing_lines]
    existing_result_ids = {line['id']: line for line in existing_lines}
wf = open(fn, "w")
all_results = []
for j, (inputs, lang_tgts, probe_outs, raw_state_targets, init_states) in enumerate(tqdm(convert_to_transformer_batches(
    dev_dataset, tokenizer, EVAL_BATCHSIZE, include_init_state=encode_init_state, domain="alchemy", device=args.device
))):
    if create_type == "drain_half":
        # only previously had 1 command
        if tokenizer.decode(inputs['input_ids'][-1], skip_special_tokens=True).count('.\n') > 0:
            continue
    encoder_outputs, input_ids, attention_mask = encode_input(tokenized_sentence=inputs)
    hidden_reps_mask = attention_mask.clone()
    orig_output = base_lm.generate(input_ids=input_ids, decoder_start_token_id=base_lm.config.pad_token_id, max_length=128, no_repeat_ngram_size=0)

    for idx, final_state in enumerate(tqdm(raw_state_targets['full_state'])):
        id = j*EVAL_BATCHSIZE + idx
        result = {'id': id}
        if id in existing_result_ids:
            all_results.append(existing_result_ids[id])
            wf.write(json.dumps(existing_result_ids[id])+"\n")
            continue
        # get final state in correct form
        orig_final_state_raw = translate_nl_to_states(final_state, "alchemy")
        orig_final_state_raw = parse_world(" ".join(i[2:] for i in orig_final_state_raw.split(" ")))
        # skip if <2 nonempty (>=2 to create the 2 sentences)
        num_nonempty = len(orig_final_state_raw['objects']) - orig_final_state_raw['objects'].count(None)
        if num_nonempty < 2:
            all_results.append(result)
            wf.write(json.dumps(result)+"\n")
            continue
        # get original utt and prior context
        orig_utt = tokenizer.decode(orig_output[idx], skip_special_tokens=True)
        priorTxt = inputs['original_text'][idx]
        assert tokenizer.decode(input_ids[idx], skip_special_tokens=True) == priorTxt.replace('  ', ' ').replace('\n', ' ')
        if '.' in priorTxt: priorTxt_rawstate = init_states[idx] + '. ' + priorTxt[priorTxt.index('.')+2:]
        else: priorTxt_rawstate = init_states[idx] + '. ' + priorTxt
        result["priorTxt"] = priorTxt_rawstate
        result["orig_utt"] = orig_utt
        orig_hidden_reps, orig_input_ids, orig_attn_mask = encoder_outputs.last_hidden_state[idx], input_ids[idx], attention_mask[idx]
        # create new sentence
        new_sentence, new_sentence_rawstate, new_final_state_raw, target_beaker_pos = create_new_sentence(
            orig_utt, priorTxt, priorTxt_rawstate, orig_final_state_raw, create_type)
        result["newTxt"] = new_sentence_rawstate
        # generate new's next utt
        new_encoder_outputs, new_input_ids, new_attention_mask = encode_input(new_sentence)
        new_utt = tokenizer.decode(base_lm.generate(
            encoder_outputs=new_encoder_outputs, encoder_attention_mask=new_attention_mask, decoder_start_token_id=base_lm.config.pad_token_id, max_length=128, no_repeat_ngram_size=0,
        )[0], skip_special_tokens=True)
        new_hidden_reps = new_encoder_outputs.last_hidden_state
        result["new_utt"] = new_utt
        if update_orig:
            priorTxt, priorTxt_rawstate, orig_final_state_raw = create_orig_sentence(
                new_utt, new_sentence, new_sentence_rawstate, new_final_state_raw,
                priorTxt, priorTxt_rawstate, orig_final_state_raw, create_type,
            )
            orig_encoder_outputs, orig_input_ids, orig_attn_mask = encode_input(priorTxt)
            result["priorTxt"] = priorTxt_rawstate
            result["orig_utt"] = tokenizer.decode(base_lm.generate(
                encoder_outputs=orig_encoder_outputs, encoder_attention_mask=orig_attn_mask, decoder_start_token_id=base_lm.config.pad_token_id, max_length=128, no_repeat_ngram_size=0,
            )[0], skip_special_tokens=True)
            orig_hidden_reps, orig_input_ids, orig_attn_mask = orig_encoder_outputs.last_hidden_state[0], orig_input_ids[0], orig_attn_mask[0]
        # replace corresponding tokens from new sentence + generate new utt
        hidden_reps_mix, attention_mask_mix = replace_tokens(
            orig_hidden_reps, orig_input_ids, orig_attn_mask, new_hidden_reps[0], new_input_ids[0], new_attention_mask[0], beaker_to_replace=target_beaker_pos)
        # get metric
        mix_encoder_outputs = ModelOutput(last_hidden_state=hidden_reps_mix.unsqueeze(0))
        orig_new_mix_utt = tokenizer.decode(base_lm.generate(
            encoder_outputs=mix_encoder_outputs, encoder_attention_mask=attention_mask_mix,
            decoder_start_token_id=base_lm.config.pad_token_id, max_length=128, no_repeat_ngram_size=0)[0], skip_special_tokens=True)
        result["mixed_utt"] = orig_new_mix_utt
        all_results.append(result)
        wf.write(json.dumps(result)+"\n")

wf.close()

f = open(fn)
all_results = [json.loads(line) for line in f]
f.close()
num_orig_consistent = 0
num_orig_consistent_new = 0
num_new_consistent_orig = 0
num_new_orig_mix_consistent = 0
num_new_consistent = 0
num_mix_consistent_if_new_consistent = 0
num_new_and_mix_same = 0
num_mix_consistent_orig_world = 0

num_consistent_w_orig_and_new = [0,0,0]

num_total = 0
unchanged_orig_utt = 0
new_new_wrong = 0
new_last_utt_same = 0
for i, result in enumerate(all_results):
    if 'priorTxt' not in result: continue
    num_total += 1
    priorTxt_rawstate = result["priorTxt"]
    orig_utt = result["orig_utt"]
    num_orig_consistent += int(consistencyCheck(priorTxt_rawstate, orig_utt))
    # if consistencyCheck(priorTxt_rawstate, orig_utt):
    new_sentence_rawstate = result["newTxt"]
    num_orig_consistent_new += int(consistencyCheck(new_sentence_rawstate, orig_utt))
    orig_new_mix_utt = result["mixed_utt"]
    # check new utt feasible under world of new sentence
    num_new_orig_mix_consistent += int(consistencyCheck(new_sentence_rawstate, orig_new_mix_utt))
    new_utt = result["new_utt"]
    num_new_consistent_orig += int(consistencyCheck(priorTxt_rawstate, new_utt))
    num_new_consistent += int(consistencyCheck(new_sentence_rawstate, new_utt))
    if consistencyCheck(new_sentence_rawstate, new_utt):
        num_mix_consistent_if_new_consistent += int(consistencyCheck(new_sentence_rawstate, orig_new_mix_utt))
    num_new_and_mix_same += int(new_utt == orig_new_mix_utt)
    # check whether new utt feasible under world of orig sentence (may or may not be the case)
    num_mix_consistent_orig_world += int(consistencyCheck(priorTxt_rawstate, orig_new_mix_utt))

    num_consistent_w_orig_and_new[0] += int(consistencyCheck(priorTxt_rawstate, orig_utt) and consistencyCheck(new_sentence_rawstate, orig_utt))
    num_consistent_w_orig_and_new[1] += int(consistencyCheck(priorTxt_rawstate, new_utt) and consistencyCheck(new_sentence_rawstate, new_utt))
    num_consistent_w_orig_and_new[2] += int(consistencyCheck(priorTxt_rawstate, orig_new_mix_utt) and consistencyCheck(new_sentence_rawstate, orig_new_mix_utt))

print("statement/world")
print(f"orig/orig: {num_orig_consistent} / {num_total} = {num_orig_consistent / num_total}")
print(f"mix/orig: {num_mix_consistent_orig_world} / {num_total} = {num_mix_consistent_orig_world / num_total}")
print(f"new/orig: {num_new_consistent_orig} / {num_total} = {num_new_consistent_orig / num_total}")
print(f"orig/new: {num_orig_consistent_new} / {num_total} = {num_orig_consistent_new / num_total}")
print(f"mix/new: {num_new_orig_mix_consistent} / {num_total} = {num_new_orig_mix_consistent / num_total}")
print(f"new/new: {num_new_consistent} / {num_total} = {num_new_consistent / num_total}")
print(f"new=mix: {num_new_and_mix_same} / {num_total} = {num_new_and_mix_same / num_total}")
print(f"(mix if new)/new: {num_mix_consistent_if_new_consistent} / {num_new_consistent} = {num_mix_consistent_if_new_consistent / num_new_consistent}")

print(np.array(num_consistent_w_orig_and_new) / num_total * 100)
