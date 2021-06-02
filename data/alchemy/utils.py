from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import itertools
from transformers.modeling_outputs import BaseModelOutput
from transformers import BartTokenizerFast, T5TokenizerFast
import sys
import numpy as np
import torch.nn.functional as F

word_to_int = {"first": 0,
    "last":6,
    "second":1,
    "third":2,
    "fourth":3,
    "fifth":4,
    "second to last": 5
}

colors = ['green', 'red', 'orange', 'purple', 'yellow', 'brown']

d_colors = {color[0]: color for color in colors}
int_to_word = {word_to_int[word]: word for word in word_to_int}
MAX_BEAKER_HEIGHT = 4


def gen_all_beaker_states(domain, args, encoding="NL", tokenizer=None, device='cuda'):
    """
    Get all possible beaker states (in state form)
    { |}the {first|second|...|last} beaker {has {1|2|3|4} {orange|red|...}{ and ...}|is empty}

    if tokenizer specified, will return tokenized states
    """
    all_states = {}
    if domain == "alchemy":
        # all beaker combinations, filled from 0-4
        all_beaker_states = set()
        for beaker_amount in range(5):
            if beaker_amount == 0: all_beaker_states.add("_")
            else: all_beaker_states = all_beaker_states.union(set(itertools.product(d_colors, repeat=beaker_amount)))
        # 1555 states = 1 + 6 + 6^2 + 6^3 + 6^4

        # can have any beaker in any of 7 positions
        all_beaker_states = {f'{i+1}:{"".join(bstate)}' for i in range(len(int_to_word)) for bstate in all_beaker_states}
        # 10885 states

        # make insensitive to separations of the same color (obo and oob are the same...)
        # (this is just to make the NL version consistent with the raw version)
        # (won't make a difference in practice as there are <= 2 colors and <= 1 color separation interface in all beakers)
        nl_all_beaker_states = set()
        raw_all_beaker_states = set()
        nl_filtered_all_beaker_states = []
        nl_beaker_state_to_idx = {}
        raw_filtered_all_beaker_states = []
        raw_beaker_state_to_idx = {}
        for bstate in all_beaker_states:
            if translate_states_to_nl(bstate, domain, not tokenizer or isinstance(tokenizer, BartTokenizerFast)) not in nl_all_beaker_states:
                nl_all_beaker_states.add(translate_states_to_nl(bstate, domain, not tokenizer or isinstance(tokenizer, BartTokenizerFast)))
                raw_all_beaker_states.add(encodeState(domain, bstate, device))
                nl_state = decide_translate(bstate, args.probe_target, domain, not tokenizer or isinstance(tokenizer, BartTokenizerFast))
                nl_beaker_state_to_idx[nl_state] = len(nl_filtered_all_beaker_states)
                nl_filtered_all_beaker_states.append(nl_state)
                raw_state = encodeState(domain, bstate, device)
                raw_beaker_state_to_idx[nl_state] = len(raw_filtered_all_beaker_states)
                raw_filtered_all_beaker_states.append(raw_state)
        if encoding == "NL":
            # 7315 states
            if not tokenizer:
                return nl_filtered_all_beaker_states, nl_beaker_state_to_idx
            else:
                tokenized_beaker_states = tokenizer(nl_filtered_all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(device)
                return tokenized_beaker_states, nl_beaker_state_to_idx
        elif encoding == "raw":
            return raw_filtered_all_beaker_states, raw_beaker_state_to_idx
    raise NotImplementedError


def gen_all_beaker_pos(domain, args, encoding="NL", tokenizer=None, device='cuda'):
    """
    Get all possible beaker positions
    { |}the {first|second|...|last} beaker

    if tokenizer specified, will return tokenized states
    """
    all_states = {}
    if domain == "alchemy" and encoding == "NL":
        # can have any beaker in any of 7 positions
        all_beaker_states = [f'{"" if i == 0 else " "}the {int_to_word[i]} beaker' for i in range(len(int_to_word))]
        beaker_state_to_idx = {all_beaker_states[i]: i for i in range(len(all_beaker_states))}
        if not tokenizer:
            return all_beaker_states, beaker_state_to_idx
        else:
            tokenized_beaker_states = tokenizer(all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(device)
            return tokenized_beaker_states, beaker_state_to_idx
    else: raise NotImplementedError


def gen_all_beaker_colors(domain, args, encoding="NL", tokenizer=None, device='cuda'):
    """
    Get all possible beaker colors (including `empty`)

    if tokenizer specified, will return tokenized states
    """
    all_states = {}
    if domain == "alchemy" and encoding == "NL":
        all_beaker_states = colors + ["empty"]
        beaker_state_to_idx = {all_beaker_states[i]: i for i in range(len(all_beaker_states))}
        if not tokenizer:
            return all_beaker_states, beaker_state_to_idx
        else:
            tokenized_beaker_states = tokenizer(all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(device)
            return tokenized_beaker_states, beaker_state_to_idx
    else: raise NotImplementedError


def gen_all_beaker_amounts(domain, args, encoding="NL", tokenizer=None, device='cuda'):
    """
    Get all possible beaker amounts (0 <= x <= 4)

    if tokenizer specified, will return tokenized states
    """
    all_states = {}
    if domain == "alchemy" and encoding == "NL":
        all_beaker_states = [f'{i if i > 0 else "empty"}' for i in range(5)]
        beaker_state_to_idx = {all_beaker_states[i]: i for i in range(len(all_beaker_states))}
        if not tokenizer:
            return all_beaker_states, beaker_state_to_idx
        else:
            tokenized_beaker_states = tokenizer(all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(device)
            return tokenized_beaker_states, beaker_state_to_idx
    else: raise NotImplementedError


def get_matching_state_labels(all_states, beaker_state_to_idx, target_states, encode_tgt_state, tokenizer, device='cuda'):
    """
    get indices of `target_states` among `all_states`
    (for creating label)

    Both are tokenized {'input_ids': torch.tensor, 'attention_mask': torch.tensor}
    """
    bs = target_states['input_ids'].size(0)

    labels = []
    for i in range(len(target_states['input_ids'])):
        if encode_tgt_state.split('.')[0] == 'NL':
            target = tokenizer.decode(target_states['input_ids'][i], skip_special_tokens=True)
        elif encode_tgt_state.split('.')[0] == 'raw':
            target = tokenizer.decode(target_states['input_ids'][i], skip_special_tokens=True)
        labels.append(beaker_state_to_idx[target])

    labels = torch.tensor(labels).to(device)
    if encode_tgt_state.split('.')[0] == 'NL':
        assert (all_states['input_ids'][labels][all_states['attention_mask'][labels].bool()] == target_states['input_ids'][target_states['attention_mask'].bool()]).all()
    return labels


def encodeState(domain, state, device='cuda'):
    state = state.split(' ')
    if domain == 'alchemy':
        assert len(state) < len(int_to_word)
        x = np.zeros([2, max(len(int_to_word), len(colors))])
        for contents in state:
            ix = int(contents.split(':')[0]) - 1
            contents = contents.split(':')[1]  #get rid of number and colon
            if contents == '_':
                height = 0	
            else:
                height = len(contents)
            for lvl in range(height):
                x[0,ix] = 1; x[1,colors.index(d_colors[contents[lvl]])] += 1
        return torch.tensor(x.flatten()).float().to(device)
    elif domain == 'scene':
        assert len(state) == NUM_POSITIONS
        x = np.zeros((NUM_POSITIONS, 2, NUM_COLORS))

        for ix, contents in enumerate(state):
            assert len(contents) == 2
            shirt, hat = contents[0], contents[1]
            x[ix, 0, COLORS_TO_INDEX[shirt]] = 1
            x[ix, 1, COLORS_TO_INDEX[hat]] = 1
        return torch.tensor(x.flatten()).float().to(device)
    else: raise NotImplementedError


def check_well_formedness(states):
    # check well-formedness of NL state
    states = states.split(', ')
    try:
        for state in states:
            if "is empty" in state:
                assert state.startswith("the ") or state.startswith(" the ")
                assert state.endswith(" beaker is empty")
                state = state.split("the ")[1].split(" beaker is empty")[0]
                assert state in word_to_int
            else:
                state = state.split(" beaker has ")
                assert len(state) == 2
                state[0] = state[0].split('the ')[1]
                assert state[0] in word_to_int
                state[1] = state[1].split(' and ')
                for color in state[1]:
                    color = color.split(' ')
                    assert len(color) == 2
                    assert int(color[0]) < 5
                    assert color[1] in colors
    except (AssertionError, ValueError, IndexError, KeyError): return False
    return True




def translate_states_to_nl(state_inputs, domain, add_space=True):
    """
    If second-last beaker, adds ' ' before the sentence
    """
    if domain == "alchemy":
        all_beakers = state_inputs.split(" ")

        nl_states = []
        for beaker_state in all_beakers:
            beaker_number = int_to_word[int(beaker_state.split(":")[0])-1]
            if '_' in beaker_state:
                nl_states.append(f"the {beaker_number} beaker is empty")
            elif len(set(beaker_state.split(":")[1])) == 1:  # only 1 color in beaker
                color_in_beaker = d_colors[beaker_state.split(":")[1][0]]
                amount_in_beaker = len(beaker_state.split(":")[1])
                nl_states.append(f"the {beaker_number} beaker has {amount_in_beaker} {color_in_beaker}")
            else:
                colors_to_amount = {}
                beaker_items = beaker_state.split(":")[1]
                for item in beaker_items:
                    if d_colors[item] not in colors_to_amount:
                        colors_to_amount[d_colors[item]] = 0
                    colors_to_amount[d_colors[item]] += 1
                string = []
                for color in colors_to_amount:
                    string.append(f"{colors_to_amount[color]} {color}")
                string = " and ".join(string)
                string = f"the {beaker_number} beaker has {string}"
                nl_states.append(string)

        if len(nl_states) == 1:
            # only 1 element--add space for non-first beakers
            nl_states = f"{' ' if beaker_number != 'first' and add_space else ''}{nl_states[0]}"
        else:
            nl_states = ", ".join(nl_states)
        return nl_states
    else: raise NotImplementedError()


def translate_nl_to_states(nl_inputs, domain):
    if domain == "alchemy":
        # non-well-formed
        if not check_well_formedness(nl_inputs): return ""
        all_beakers = nl_inputs.split(", ")
        raw_states = []
        for beaker_state in all_beakers:
            if "is" in beaker_state: beaker_state = beaker_state.split(" is ")
            elif "has" in beaker_state: beaker_state = beaker_state.split(" has ")
            else: assert False
            beaker_number = word_to_int[beaker_state[0].split("the ")[1].split(" beaker")[0]] + 1
            beaker_contents = beaker_state[1]
            if 'empty' in beaker_state:
                raw_states.append(f"{beaker_number}:_")
            elif "and" not in beaker_contents:  # only 1 color in beaker
                beaker_contents = beaker_contents.split(' ')
                amount = int(beaker_contents[0])
                color = beaker_contents[1][0]
                raw_states.append(f"{beaker_number}:{''.join([color for _ in range(amount)])}")
            else:
                string = ""
                beaker_contents = beaker_contents.split(" and ")
                for item in beaker_contents:
                    item = item.split(" ")
                    amount = int(item[0])
                    color = item[1][0]
                    for _ in range(amount): string += color
                string = f"{beaker_number}:{string}"
                raw_states.append(string)
        raw_states = " ".join(raw_states)
        return raw_states
    else: raise NotImplementedError()


def decide_translate(probe_target, state_targets_type, domain, add_space=True):
    # probe_target: 'state', 'init_state', 'interm_states', 'single_beaker', 'text'
    state_targets_type = state_targets_type.split('.')
    if len(state_targets_type) > 1 and state_targets_type[-1] == 'NL': return translate_states_to_nl(probe_target, domain, add_space=add_space)
    else: return probe_target
