from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from data.alchemy.alchemy_artificial_generator import execute
from data.alchemy.parse_alchemy import (
    consistencyCheck, parse_utt_with_world, parse_world,
)
from data.alchemy.parseScone import getBatchesWithInit
from data.alchemy.utils import (
    check_well_formedness, word_to_int, colors, d_colors as char_to_color, translate_states_to_nl,
)
import itertools
import Levenshtein
from transformers.modeling_outputs import BaseModelOutput
import sys
import numpy as np
import torch.nn.functional as F


# how similar are the states
def get_state_similarity(orig_state1, orig_state2, domain, target):
    '''
    converts states to form:
        the first beaker has 3 green, the second beaker has 1 yellow and 1 green, ... // 1:ggg 2:yg
            ==>>
        ['green', 'red', 'orange', 'purple', 'yellow', 'brown']
        3g,1y/1g,2g/1y/1p
        3y,2y/1g,1g/2p
        Levenshtein for *each* beaker

        gg-- vs. yy-- should be same penalty as g--- vs. y---
        (?) 4g vs. 1g should be penalized harder than 2g vs. 1g (but 1 action can be `pour 3`?)
            difference in amounts (1-3) + wrong color (2)
        
        right amount (+1) + right color (+1)
        if >= 2 colors in beaker:
            how many colors correct (+1/color) + is each amount correct (+1/amount)
    then compares them
    '''
    def translate_beaker_nl_to_state_list(state_descr):
        '''
        converts states to form:
            the first beaker has 3 green, the second beaker has 1 yellow and 1 green, ...
                ==>>
            [['1', '3g'], ['2', '1y', '1g']] if 'single_beaker' in target
            [['3g'], ['1y', '1g']] if 'single_beaker' not in target

            ['green', 'red', 'orange', 'purple', 'yellow', 'brown']
            [[3,0,0,0,0,0], [1,0,0,0,1,0]]
        '''
        state_descr = state_descr.split(', ')
        state_list = []
        for beaker in state_descr:
            beaker_contents = []
            if "has" in beaker:  # nonempty
                beaker = beaker.split(' beaker has ')
            elif "is" in beaker:  # empty
                beaker = beaker.split(' beaker is ')
            else: assert False
            bn = beaker[0].split('the ')[1]
            beaker_colors = beaker[1].split(' and ')
            for beaker_color in beaker_colors:
                if beaker_color == "empty":
                    amount = 0
                    beaker_contents.append(f'{amount}')
                else:
                    beaker_color = beaker_color.split(' ')
                    amount = beaker_color[0]
                    colour = beaker_color[1][0]
                    beaker_contents.append(f"{amount}{colour}")
            beaker_contents = list(set(beaker_contents))
            if 'single_beaker' in target:
                # add positional info
                beaker_contents.insert(0, str(word_to_int[bn]))
            state_list.append([beaker_contents[0]] + list(set(beaker_contents[1:])))
        return state_list

    try:
        if not "the" in orig_state1:
            orig_state1 = translate_states_to_nl(orig_state1, domain)
        if not "the" in orig_state2:
            orig_state2 = translate_states_to_nl(orig_state2, domain)
    except (ValueError, KeyError, IndexError):
        # non-well-formed inputs
        return 0.0
    if not check_well_formedness(orig_state1) or not check_well_formedness(orig_state2): return 0.0
    state1 = translate_beaker_nl_to_state_list(orig_state1)
    state2 = translate_beaker_nl_to_state_list(orig_state2)

    # compare state1 and state2
    n_match = 0.0
    n_total = 0.0
    for i in range(max(len(state1), len(state2))):
        max_similarity = 0
        if i < min(len(state1), len(state2)):
            if 'single_beaker' in target: num_pos_tokens = 1
            else: num_pos_tokens = 0
            if len(state1[i]) == num_pos_tokens + 1 or len(state2[i]) == num_pos_tokens + 1:
                state1[i] = ''.join(state1[i])
                state2[i] = ''.join(state2[i])
                max_similarity = Levenshtein.ratio(state1[i], state2[i])
            else:
                state1[i] = ''.join(state1[i])
                # find permutation of colors which maximizes L-ratio w/ state1
                state2_color_pms = itertools.permutations(state2[i][num_pos_tokens:])
                for s2_perm in state2_color_pms:
                    s2_perm = state2[i][:num_pos_tokens] + list(s2_perm)
                    max_similarity = max(max_similarity, Levenshtein.ratio(state1[i], ''.join(s2_perm)))
        n_match += max_similarity
        n_total += 1.0
    return n_match / n_total


def check_val_consistency(
    model, tokenizer, inputs, lang_tgts, init_states=None, included_init_state=False, return_texts=False, generated=None,
):
    """
    extra: {"init_states": init_states, "raw_lang_tgt": lang_tgt_enc, "raw_state_tgt": state_targets}
    """
    if generated is None:
        generated = model.generate(inputs['input_ids'], decoder_start_token_id=model.config.pad_token_id, max_length=50) #model.config.decoder.pad_token_id
    n_consistent = 0
    # init_states = extra['init_states']
    if included_init_state:
        assert len(init_states) == len(generated)

    prior_list = []
    gt_list = []
    gen_list = []
    consistent_list = []

    for i in range(len(generated)):
        priorTxt = tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
        if included_init_state == 'NL':
            prior_utts = priorTxt[priorTxt.index('.')+2:]
            priorTxt_rawstate = init_states[i] + '. ' + prior_utts
        elif not included_init_state:
            priorTxt_rawstate = init_states[i] + '. ' + priorTxt
        elif included_init_state == "raw":
            priorTxt_rawstate = priorTxt
        else: assert False
        gtTxt = tokenizer.decode(lang_tgts['input_ids'][i], skip_special_tokens=True)
        if generated is not None:
            genTxt = tokenizer.decode(generated[i], skip_special_tokens=True)
        else:
            genTxt = generated[i]

        assert consistencyCheck(priorTxt_rawstate, gtTxt)
        gen_consistent = consistencyCheck(priorTxt_rawstate, genTxt)
        n_consistent += gen_consistent

        prior_list.append(priorTxt)
        gt_list.append(gtTxt)
        gen_list.append(genTxt)
        consistent_list.append(gen_consistent)
    if return_texts:
        return n_consistent, {'prior': prior_list, 'gold': gt_list, 'gen': gen_list, 'consistent': consistent_list}
    return n_consistent