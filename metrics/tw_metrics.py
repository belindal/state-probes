import json
import logging
import os
import torch
from tqdm import tqdm
import textworld
from textworld.logic import parser, State
from textworld.logic import Signature, Proposition, Action, Variable, Type
import torch.nn.functional as F
from torch import nn


def consistencyCheck(prev_context, next_utt, env, tokenizer, has_action=None, has_response=None):
    game_state = env.reset()
    prev_utts = prev_context.split('\n')
    prev_utts = [utt for utt in prev_utts if utt.startswith('>')]
    for utt in prev_utts:
        prev_game_state, reward, done = env.step(utt[2:].strip())

    next_utt = next_utt.split('[')[0]
    if has_action is None:
        has_action = (next_utt[:2] == '> ')
    # automatically extract action and/or response
    if has_action:
        next_utt = next_utt.strip().split('\n')
        if has_response is None: has_response = len(next_utt) >= 2
        # if len(next_utt) < 2: return False, True, True, "invalid"
        action = next_utt[0]
        if has_response: pred_response = '\n'.join(next_utt[1:])
        game_state, reward, done = env.step(action[2:].strip())
    else:
        # is response
        has_response = True
        assert '> ' in prev_context
        action = prev_context[prev_context.rfind('> '):].split('\n')[0]
        pred_response = next_utt
        game_state = prev_game_state

    gt_response = '\n'.join([
        feedback_line for feedback_line in game_state.feedback.strip().split('\n')
        if 'Your score has just gone up by ' not in feedback_line and len(feedback_line.strip()) > 0
    ])
    gt_response_full = gt_response
    if has_action: is_invalid = (action[2:] not in prev_game_state.admissible_commands)
    else: is_invalid = None
    if action.startswith('> go ') and not is_invalid and has_response:
        # only get the room
        gt_response = gt_response.split('\n')[0]
        pred_response = pred_response.split('\n')[0]
        try: assert gt_response.startswith('-=') and gt_response.endswith('=-')
        except AssertionError:
            assert gt_response == "You can't go that way."
            is_invalid = True
    if has_response:
        is_consistent = (gt_response.strip() == pred_response.strip())
        is_wrong_feedback = not is_consistent and not is_invalid
    else:
        is_consistent = None
        is_wrong_feedback = None
    return is_consistent, is_invalid, is_wrong_feedback, gt_response_full


def get_em(state1, state2, state_type):
    # order-invariant EM calculation
    if 'facts' in state_type:
        state1 = set(state1.split(' [SEP] '))
        state2 = set(state2.split(' [SEP] '))
        n_intersect = len(state1.intersection(state2))
        p = n_intersect / len(state1) if len(state1) != 0 else 1
        r = n_intersect / len(state2) if len(state2) != 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
        return state1 == state2, f1, state1, state2
    elif state_type == 'inventory':
        state1 = state1.replace(' an ', ' ').replace(' a ', ' ')
        state2 = state2.replace(' an ', ' ').replace(' a ', ' ')
        if not state1.startswith('You are carrying') or not state2.startswith('You are carrying'):
            # not in correct format
            return False, 0, state1, state2
        state1_set, state2_set = translate_inv_str_to_items(state1), translate_inv_str_to_items(state2)
        # sanity check
        assert translate_inv_items_to_str(state1_set) == state1 and translate_inv_items_to_str(state2_set) == state2
        state1_set, state2_set = set(state1_set), set(state2_set)
        n_intersect = len(state1_set.intersection(state2_set))
        p = n_intersect / len(state1_set) if len(state1_set) != 0 else 1
        r = n_intersect / len(state2_set) if len(state2_set) != 0 else 1
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        return state1_set == state2_set, f1, state1_set, state2_set
    else: assert False


def get_confusion_matrix(pred, label, n_labels):
    # labels are 0...(n_labels - 1)
    # pred (B x N)
    # label (B x N)
    def n_intersect(p_value, l_value):
        return ((pred == p_value) & (label == l_value)).sum(1)
    return [[n_intersect(i, j).tolist() for i in range(n_labels)] for j in range(n_labels)]

