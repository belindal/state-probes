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
import itertools
from data.textworld.utils import EntitySet


def translate_inv_items_to_str(inv_items):
    # returns string description of inventory, given a collection of items
    inv_items = list(inv_items)
    if len(inv_items) == 0: return "You are carrying nothing."
    elif len(inv_items) == 1: inv_str = f"You are carrying: {inv_items[0]}."
    elif len(inv_items) >= 2:
        inv_str = f"You are carrying: {', '.join(inv_items[:-1])}"
        inv_str += f' and {inv_items[-1]}.'
    assert len(inv_items) == inv_str.count(', ') + inv_str.count(' and ') + 1
    return inv_str


def translate_inv_str_to_items(inv_str):
    # returns list of items in inventory, given a string description
    if inv_str == 'You are carrying nothing.': return []
    assert inv_str.startswith('You are carrying: ')
    inv_items1 = inv_str.replace('You are carrying: ', '').rstrip('.').split(' and ')
    assert len(inv_items1) <= 2
    inv_items = inv_items1[0].split(', ')
    if len(inv_items1) == 2: inv_items.append(inv_items1[1])
    assert len(inv_items) == inv_str.count(', ') + inv_str.count(' and ') + 1
    return inv_items


def parse_facts_to_nl(facts, inform7_game, get_orig=False):
    # convert list of facts to nl
    nl_facts = []
    orig_facts = []
    nl_facts_set = set()
    for fact in facts:
        # check if already in NL form
        if type(fact) == str: nl_fact = fact
        else:
            fact = Proposition.deserialize(fact)
            nl_fact = inform7_game.gen_source_for_attribute(fact)
            # ensure no repeats
            if nl_fact in nl_facts_set: continue
            if len(nl_fact) == 0:
                # TODO what will we do about these?
                assert fact.name == 'free' or fact.name == 'link'
                continue
        nl_facts.append(nl_fact)
        orig_facts.append(fact)
        nl_facts_set.add(nl_fact)
    if get_orig:
        return nl_facts, orig_facts
    else:
        return nl_facts


import re
def parse_nl_to_facts(nl_facts, game_state, gameid, get_orig=False, cached_templates=None, inform7_game=None):
    facts = []
    invalid_syntax_facts = []
    predicates = game_state['game'].kb.inform7_predicates
    var_names = game_state['game'].kb.inform7_variables
    game_types = game_state['game'].kb.types
    game_ents = game_state.game.infos
    game_type_to_entities = {t: [
        game_ents[e].name for e in game_ents if game_types.is_descendant_of(game_ents[e].type, t)  # game_ents[e].type == t  #
    ] for t in game_types.types}
    entity_name_to_type = {game_ents[e].name: game_ents[e].type for e in game_ents}
    game_types = game_types.types + ["r'"]
    game_type_to_entities["r'"] = game_type_to_entities['r']
    game_type_to_entities['I'] = ['inventory']
    game_type_to_entities['P'] = ['player']
    entity_name_to_type['P'] = 'P'
    entity_name_to_type['I'] = 'I'
    # match fact to predicate
    # over all predicates

    if not cached_templates: cached_templates = {}
    if gameid not in cached_templates:
        cached_templates[gameid] = {}
        entity = re.compile("{"+f"({'|'.join(game_types)})"+"}")
        for signature in predicates:
            nl_template = predicates[signature][1]
            if len(nl_template) == 0: continue
            param_indices = [[m.start(),m.end()] for m in re.finditer(entity, nl_template)]
            symbol_to_type = {param.name: param.type for param in predicates[signature][0].parameters}
            for p in range(len(param_indices)-1,-1,-1):
                param_index_pair = param_indices[p]
                ent_type = symbol_to_type[nl_template[param_index_pair[0]+1:param_index_pair[1]-1]]
                nl_template = f"{nl_template[:param_index_pair[0]]}({'|'.join(game_type_to_entities[ent_type])}){nl_template[param_index_pair[1]:]}"
            regex = re.compile(nl_template)
            param_idx_to_nl_position = []  # param idx in predicate -> position (group #) in nl string
            for param in predicates[signature][0].parameters:
                if param.name != 'I' and param.name != 'P':
                    param_idx = predicates[signature][1].find("{"+param.type+"}")
                    param_idx_to_nl_position.append(param_idx)
            param_idx_to_nl_position = [i[0] for i in sorted(enumerate(param_idx_to_nl_position), key=lambda x:x[1])]
            if regex in cached_templates[gameid]:
                assert param_idx_to_nl_position == cached_templates[gameid][regex]['param_idx_to_nl_position']
                assert predicates[signature][0].name == cached_templates[gameid][regex]['predicate'].name
                assert predicates[signature][0].parameters[0].name == cached_templates[gameid][regex]['predicate'].parameters[0].name
                if len(predicates[signature][0].parameters) > 1:
                    assert predicates[signature][0].parameters[1].name == cached_templates[gameid][regex]['predicate'].parameters[1].name
            cached_templates[gameid][regex] = {
                'predicate': predicates[signature][0],
                'param_idx_to_nl_position': param_idx_to_nl_position,
            }
            if "The player carries" in nl_template:
                nl_template = f"T{nl_template.replace('The player carries t', '')} is in the inventory"
                regex = re.compile(nl_template)
                param_idx_to_nl_position = [0]
                cached_templates[gameid][regex] = {
                    'predicate': predicates[signature][0],
                    'param_idx_to_nl_position': param_idx_to_nl_position,
                }

    for nl_fact in nl_facts:
        # regex match template
        for matched_template in cached_templates[gameid]:
            if re.fullmatch(matched_template, nl_fact): break
        if not re.fullmatch(matched_template, nl_fact):
            invalid_syntax_facts.append(nl_fact)
            continue
        num_nonconstant_params = 0
        mapping = {}  # placeholder -> variable
        param_idx = 0
        for param in cached_templates[gameid][matched_template]['predicate'].parameters:
            if param.name != 'I' and param.name != 'P':
                entity_name = re.search(matched_template, nl_fact).group(cached_templates[gameid][matched_template]['param_idx_to_nl_position'][param_idx]+1)
                mapping[param] = Variable(entity_name, entity_name_to_type[entity_name])
                param_idx += 1
            else:
                mapping[param] = Variable(param.name, entity_name_to_type[param.name])
        fact = Proposition.serialize(cached_templates[gameid][matched_template]['predicate'].instantiate(mapping))
        facts.append(fact)
    test_nl_facts = set(parse_facts_to_nl(facts, inform7_game, get_orig=False))
    if len(test_nl_facts) == len(facts):
        try: assert test_nl_facts == set(nl_facts) - set(invalid_syntax_facts)
        except AssertionError:
            test_nl_facts = {fact if 'The player carries' not in fact else f'T{fact.replace("The player carries t", "")} is in the inventory' for fact in test_nl_facts}
            assert test_nl_facts == set(nl_facts) - set(invalid_syntax_facts)
    return facts, cached_templates

