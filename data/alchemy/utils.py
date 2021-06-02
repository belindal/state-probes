from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import itertools
import Levenshtein
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


def gen_all_beaker_states(domain, args, encoding="NL", tokenizer=None):
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
                raw_all_beaker_states.add(encodeState(domain, bstate))
                nl_state = decide_translate(bstate, args.probe_target, domain, not tokenizer or isinstance(tokenizer, BartTokenizerFast))
                nl_beaker_state_to_idx[nl_state] = len(nl_filtered_all_beaker_states)
                nl_filtered_all_beaker_states.append(nl_state)
                raw_state = encodeState(domain, bstate)
                raw_beaker_state_to_idx[nl_state] = len(raw_filtered_all_beaker_states)
                raw_filtered_all_beaker_states.append(raw_state)
        if encoding == "NL":
            # 7315 states
            if not tokenizer:
                return nl_filtered_all_beaker_states, nl_beaker_state_to_idx
            else:
                tokenized_beaker_states = tokenizer(nl_filtered_all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                return tokenized_beaker_states, nl_beaker_state_to_idx
        elif encoding == "raw":
            return raw_filtered_all_beaker_states, raw_beaker_state_to_idx
    raise NotImplementedError


def gen_all_beaker_pos(domain, args, encoding="NL", tokenizer=None):
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
            tokenized_beaker_states = tokenizer(all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            return tokenized_beaker_states, beaker_state_to_idx
    else: raise NotImplementedError


def gen_all_beaker_colors(domain, args, encoding="NL", tokenizer=None):
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
            tokenized_beaker_states = tokenizer(all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            return tokenized_beaker_states, beaker_state_to_idx
    else: raise NotImplementedError


def gen_all_beaker_amounts(domain, args, encoding="NL", tokenizer=None):
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
            tokenized_beaker_states = tokenizer(all_beaker_states, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            return tokenized_beaker_states, beaker_state_to_idx
    else: raise NotImplementedError


def encodeState(domain, state):
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
        return torch.tensor(x.flatten()).float().to(DEVICE)
    elif domain == 'scene':
        assert len(state) == NUM_POSITIONS
        x = np.zeros((NUM_POSITIONS, 2, NUM_COLORS))

        for ix, contents in enumerate(state):
            assert len(contents) == 2
            shirt, hat = contents[0], contents[1]
            x[ix, 0, COLORS_TO_INDEX[shirt]] = 1
            x[ix, 1, COLORS_TO_INDEX[hat]] = 1
        return torch.tensor(x.flatten()).float().to(DEVICE)
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
