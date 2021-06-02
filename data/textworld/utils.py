from typing import Iterable

import torch
from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader, Dataset, IterableDataset
import glob
import itertools
from transformers import PreTrainedTokenizerBase
import regex as re
import textworld


class EntitySet:
    def __init__(self, ent_list: Iterable):
        self.ent_list = sorted(ent_list, key=str)
        self.entity_set = set(ent_list)
        self.has_none = None in self.entity_set
        self.nonNone_ent = None
        for i in range(len(self.ent_list)):
            if self.ent_list[i] is not None:
                self.nonNone_ent = self.ent_list[i]
        self.has_nonNone = self.nonNone_ent is not None
    
    def __getitem__(self, i):
        return self.ent_list[i]

    def __hash__(self):
        # order invariant
        set_hash = 0
        for item in self.entity_set:
            set_hash += hash(item)
        return set_hash
    
    def __eq__(self, other):
        return self.entity_set == other.entity_set

    def __str__(self):
        return str(self.ent_list)
    
    def __len__(self):
        return len(self.entity_set)
    
    @staticmethod
    def serialize(entset):
        return json.dumps(entset.ent_list)
    
    @staticmethod
    def deserialize(string: str):
        entity_set = json.loads(string)
        return EntitySet(entity_set)


def pad_stack(inputs, pad_idx=1, device='cpu'):
    # inputs: ['input_ids', 'attention_mask']: list of tensors, of dim (#facts, seqlen, *)
    input_seqlens = torch.cat([inp['attention_mask'].sum(1) for inp in inputs])
    max_seqlen = input_seqlens.max()
    max_nfacts = max([inp['attention_mask'].size(0) for inp in inputs])
    input_list = []
    mask_list = []
    for i, inp in enumerate(inputs):
        mask_size = list(inp['attention_mask'].size())
        mask_size[1] = max_seqlen - mask_size[1]
        new_mask = torch.cat([inp['attention_mask'], torch.zeros(*mask_size).to(inp['attention_mask'].device, inp['attention_mask'].dtype)], dim=1)
        mask_size[0] =  max_nfacts - mask_size[0]
        new_mask = torch.cat([new_mask, torch.zeros(mask_size[0], max_seqlen, *mask_size[2:]).to(inp['attention_mask'].device, inp['attention_mask'].dtype)], dim=0)
        new_inp = torch.ones(max_nfacts, max_seqlen, *inp['input_ids'].size()[2:]).to(inp['input_ids'].device, inp['input_ids'].dtype) * pad_idx
        new_inp[new_mask.bool()] = inp['input_ids'][inp['attention_mask'].bool()]
        # pad and stack tensors
        mask_list.append(new_mask)
        input_list.append(new_inp)
    return torch.stack(input_list).to(device), torch.stack(mask_list).to(device)


def get_relevant_facts_about(entities, facts, curr_world=None, entity=None, excluded_entities=None, exact_arg_count=True, exact_arg_order=False):
    '''
    entities: list of entities that should *all* appear in list of facts to get (except for the `None` elements)
    excluded_entities: list of entities that should *never* appear in list of facts to get (overrides `entities`)
    exact_arg_count: only get facts with the exact # of non-None arguments as passed-in `entities`
    exact_arg_order: only get facts with the exact order of non-None arguments as passed-in `entities`
    '''
    relevant_facts = []
    count_nonNone_entities = len([e for e in entities if e is not None])
    for fact in facts:
        exclude_fact = False
        fact_argnames = [arg['name'] for arg in fact['arguments']]
        if exact_arg_count and len(fact_argnames) != count_nonNone_entities: continue  # argument count doesn't match
        if "I" in fact_argnames: fact_argnames[fact_argnames.index("I")] = "inventory"
        if "P" in fact_argnames: fact_argnames[fact_argnames.index("P")] = "player"
        # check none of `excl_entity` shows up in `fact`
        if excluded_entities is not None:
            for excl_entity in excluded_entities:
                # if excl_entity == 'P' or excl_entity == 'I': excl_entity = 'player'
                if excl_entity in fact_argnames:
                    exclude_fact = True
                    break
        if exclude_fact: continue
        add_fact = True
        # if exact_arg_order, entities must appear in (correct position of) fact
        # otherwise, entities must appear (anywhere) in fact
        for e, entity in enumerate(entities):
            if entity is not None and ((exact_arg_order and entity != fact_argnames[e]) or (not exact_arg_order and entity not in fact_argnames)):
                add_fact = False
                continue
        if add_fact: relevant_facts.append(fact)
    return relevant_facts


def load_possible_pairs(pair_out_file=None, data_dir=None, game_ids=None):
    if pair_out_file and os.path.exists(pair_out_file):
        possible_pairs_serialized = json.load(open(pair_out_file))
        # deserialize
        possible_pairs = {}
        for gameid in possible_pairs_serialized:
            possible_pairs[gameid] = [EntitySet.deserialize(pair_str) for pair_str in possible_pairs_serialized[gameid]]
        return possible_pairs
    elif game_ids is not None:
        return gen_possible_pairs(data_dir, game_ids)[0]
    else:
        return None


def load_negative_tgts(negative_tgts_fn=None, data_dir=None, ent_set_size=None, tokenizer=None, game_ids=None):
    if negative_tgts_fn and os.path.exists(negative_tgts_fn):
        negative_tgts_serialized = torch.load(negative_tgts_fn)
        return negative_tgts_serialized
    elif game_ids is not None:
        return gen_all_facts(data_dir, state_encoder, None, tokenizer, game_ids, ent_set_size)
    else:
        return None


def apply_mask_and_truncate(tensor, mask, max_len, device):
    """
    tensor (bsz, seqlen, *)
    mask (bsz)
    max_len (int)
    """
    return tensor[mask][:,:max_len].to(device)


def remap_entset(entset, control_mapping):
    """
    Transform entities of entity set according to `control_mapping`
    {ent1, ent2, ...} -> {control_mapping[ent1], control_mapping[ent2], ...}
    """
    remapped_entset = [None for _ in entset]
    # create (possibly control) names of mentions...
    for e, entity in enumerate(entset):
        # if control task, transform mentions...
        entity_name = entity
        if entity is not None:
            entity_name = control_mapping[entity_name] if entity_name in control_mapping else entity_name
        remapped_entset[e] = entity_name
    return remapped_entset


def gen_possible_pairs(data_dir, game_ids):
    print("Getting all entity types")
    # from fact and entity types
    type_to_gid_to_ents = {}
    gameid_to_state = {}
    for game_id in tqdm(game_ids):
        env = textworld.start(os.path.join(data_dir, f'{game_id}.ulx'))
        game_state = env.reset()
        gameid_to_state[game_id] = game_state
        game_ents = game_state.game.infos
        # filter only the types we can store in the inventory...
        game_types = game_state.game.kb.types
        type_to_gid_to_ents[game_id] = {}
        for t in game_types.types:
            if t not in type_to_gid_to_ents[game_id]: type_to_gid_to_ents[game_id][t] = []
            type_to_gid_to_ents[game_id][t] += [game_ents[e].name for e in game_ents if game_types.is_descendant_of(game_ents[e].type, t)]
        type_to_gid_to_ents[game_id]['I'] = ['inventory']
        type_to_gid_to_ents[game_id]['P'] = ['player']

    print("Getting all possible pairs")
    all_possible_pairs = {}
    type_pairs = {}
    for game_id in tqdm(game_ids):
        predicates = gameid_to_state[game_id]['game'].kb.inform7_predicates
        var_names = gameid_to_state[game_id]['game'].kb.inform7_variables
        if game_id not in all_possible_pairs:
            all_possible_pairs[game_id] = set()
            type_pairs[game_id] = set()
        # over all predicates
        for signature in predicates:
            if len(predicates[signature][1]) == 0: continue
            if signature.types in type_pairs: continue
            type_pairs[game_id].add(signature.types)
            obj_pairs = set(itertools.product(type_to_gid_to_ents[game_id][signature.types[0]], type_to_gid_to_ents[game_id][signature.types[1]])) if len(signature.types) == 2 else set(itertools.product(type_to_gid_to_ents[game_id][signature.types[0]], [None]))
            obj_pairs = list(obj_pairs)
            for p, pair in enumerate(obj_pairs):
                obj_pairs[p] = EntitySet(pair)
            all_possible_pairs[game_id] = all_possible_pairs[game_id].union(obj_pairs)
        all_possible_pairs[game_id] = list(all_possible_pairs[game_id])
    return all_possible_pairs, type_to_gid_to_ents


def gen_all_facts(gamefile, state_encoder, probe_outs, tokenizer, game_ids, ent_set_size, device):
    game_id_to_entities = {}
    game_id_to_objs = {}
    game_ids_to_kb = {}
    for game_id in game_ids:
        if game_id not in game_id_to_entities:
            env = textworld.start(os.path.join(gamefile, f'{game_id}.ulx'))
            game_state = env.reset()
            game_ents = game_state.game.infos
            # filter only the types we can store in the inventory...
            game_types = game_state.game.kb.types
            game_ids_to_kb[game_id] = game_state['game'].kb
            game_id_to_entities[game_id] = {
                t: [game_ents[e].name for e in game_ents if game_types.is_descendant_of(game_ents[e].type, t)] for t in game_types.types
            }
            game_id_to_objs[game_id] = game_id_to_entities[game_id]['o']
    if ent_set_size == 2:
        return gen_all_facts_pairs(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb, device)
    elif ent_set_size == 1:
        return gen_all_facts_single(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb, device)
    else:
        raise AssertionError


def gen_all_facts_single(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb, device):
    SPLIT_SIZE = 128
    fact_to_template = {}
    all_facts = []
    entity_name_to_types = {}
    for game_id in game_ids:
        predicates = game_ids_to_kb[game_id].inform7_predicates
        var_names = game_ids_to_kb[game_id].inform7_variables
        # over all predicates
        for signature in predicates:
            if len(predicates[signature][1]) == 0: continue
            # skip 3-arg facts (will never be `true` anyway)
            if len(predicates[signature][0].parameters) > 2: continue
            if predicates[signature][1].count('{') > 1:
                # all combinations to fill in the blanks, with at least one 'entity':
                # (r, r, o) -> ('entity', {r0...r5}, {o0...o7}); ({r0...r5}, 'entity', {o0...o7}); ({r0...r5}, {r0...r5}, 'entity');
                # over which blank we fill with `entity`
                blanks_possibles = []
                for b in range(len(signature.types)):
                    blanks_possibles.append([
                        ['entity'] if b2 == b else game_id_to_entities[game_id][typ]
                        for b2, typ in enumerate(signature.types) #if game_ids_to_kb[game_id].types.is_constant(typ)
                    ])
                    for arg_combines in itertools.product(*blanks_possibles[b]):
                        pred_str = predicates[signature][1]
                        for a, arg in enumerate(arg_combines):
                            arg_template_name = predicates[signature][0].parameters[a].name
                            assert "{"+arg_template_name+"}" in pred_str
                            pred_str = pred_str.replace('{'+arg_template_name+'}', arg)
                        fact_to_template[pred_str] = signature
                        all_facts.append(pred_str)
            else:
                pred_str = predicates[signature][1]
                for t, typ in enumerate(signature.types):  # necessary to delete the type-specific templates
                    # replace non-constants
                    if not game_ids_to_kb[game_id].types.is_constant(typ): break
                typ_name = predicates[signature][0].parameters[t].name
                assert predicates[signature][0].parameters[t].type == typ
                assert "{"+typ_name+"}" in pred_str
                pred_str = pred_str.replace("{"+typ_name+"}", "entity")
                fact_to_template[pred_str] = signature
                all_facts.append(pred_str)
                assert pred_str.count('{') == 0
        for typ in game_id_to_entities[game_id]:
            if typ == "I": entity_name_to_types["inventory"] = {"I"}; continue
            if typ == "P": entity_name_to_types["player"] = {"P"}; continue
            for name in game_id_to_entities[game_id][typ]:
                if name is not None:
                    if name not in entity_name_to_types: entity_name_to_types[name] = set()
                    if '-=' in name: name = name[3:-3].lower()
                    entity_name_to_types[name].add(var_names[typ])
    all_facts = list(set(all_facts))

    if probe_outs is None: probe_outs = {}
    probe_outs['e_name_to_types'] = entity_name_to_types

    '''
    create entity-specific targets
    '''
    # ([bs *] # facts, seqlen): [facts about entity 0, facts about entity 1, etc.]
    probe_outs['all_entity_vectors'] = {}  # fill in entities
    probe_outs['all_entity_inputs'] = {}
    probe_outs['state_to_idx'] = {}
    probe_outs['idx_to_state'] = {}
    for entity in tqdm(entity_name_to_types):
        entset_serialize = EntitySet.serialize(EntitySet([entity]))
        probe_outs['idx_to_state'][entset_serialize] = []
        facts_unique = set()
        for fact in all_facts:
            if "-= " in entity: entity = entity[3:-3].lower()
            old_fact = fact
            fact = fact.replace('entity', entity)
            if fact not in facts_unique:
                probe_outs['idx_to_state'][entset_serialize].append(fact)
                facts_unique.add(fact)
                fact_to_template[fact] = fact_to_template[old_fact]
        probe_outs['idx_to_state'][entset_serialize] = list(set(probe_outs['idx_to_state'][entset_serialize]))
        probe_outs['state_to_idx'][entset_serialize] = {fact: i for i, fact in enumerate(probe_outs['idx_to_state'][entset_serialize])}
        # input_ids: ([bs *] # facts, seqlen), attention_mask: ([bs *] # facts, seqlen)
        probe_outs['all_entity_inputs'][entset_serialize] = tokenizer(probe_outs['idx_to_state'][entset_serialize], return_tensors='pt', padding=True, truncation=True).to(device)

        encoded_inputs = []
        # save memory
        for split in range(0, probe_outs['all_entity_inputs'][entset_serialize]['input_ids'].size(0),SPLIT_SIZE):
            inp_ids = probe_outs['all_entity_inputs'][entset_serialize]['input_ids'][split:split+SPLIT_SIZE]
            attn_mask = probe_outs['all_entity_inputs'][entset_serialize]['attention_mask'][split:split+SPLIT_SIZE]
            encoded_inputs.append(state_encoder(input_ids=inp_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state.to('cpu'))
        encoded_inputs = torch.cat(encoded_inputs)

        '''
        model forward
        '''
        # encode everything
        # (bs * # facts, seqlen, embeddim)
        probe_outs['all_entity_vectors'][entset_serialize] = {
            'input_ids': encoded_inputs,
            'attention_mask': probe_outs['all_entity_inputs'][entset_serialize]['attention_mask'].to('cpu'),
        }
        probe_outs['all_entity_inputs'][entset_serialize].to('cpu')
    probe_outs['fact_to_template'] = fact_to_template
    return probe_outs


def gen_all_facts_pairs(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb, device):
    # entity -> all possible facts pertaining to that entity
    # entities: list of batch's entities to probe for...
    # get all containers
    all_facts = []
    entity_name_to_types = {}
    for game_id in game_ids:
        predicates = game_ids_to_kb[game_id].inform7_predicates
        var_names = game_ids_to_kb[game_id].inform7_variables
        # over all predicates
        for signature in predicates:
            if len(predicates[signature][1]) == 0: continue
            pred_str = predicates[signature][1]
            for t, typ in enumerate(signature.types):  # necessary to delete the type-specific templates
                # replace non-constants
                if not game_ids_to_kb[game_id].types.is_constant(typ):
                    typ_name = predicates[signature][0].parameters[t].name
                    assert predicates[signature][0].parameters[t].type == typ
                    assert "{"+typ_name+"}" in pred_str
                    pred_str = pred_str.replace("{"+typ_name+"}", f"entity{t}")
            all_facts.append(pred_str)
            assert pred_str.count('{') == 0
        for typ in game_id_to_entities[game_id]:
            if typ == "I": entity_name_to_types["inventory"] = {"I"}; continue
            if typ == "P": entity_name_to_types["player"] = {"P"}; continue
            for name in game_id_to_entities[game_id][typ]:
                if name is not None:
                    if name not in entity_name_to_types: entity_name_to_types[name] = set()
                    if '-=' in name: name = name[3:-3].lower()
                    entity_name_to_types[name].add(var_names[typ])
    all_facts = list(set(all_facts))
    # tokenize
    fact_to_idx = {fact: i for i, fact in enumerate(all_facts)}

    if probe_outs is None: probe_outs = {}
    # probe_outs['state_to_idx'] = fact_to_idx
    probe_outs['idx_to_state'] = all_facts
    probe_outs['e_name_to_types'] = entity_name_to_types

    '''
    create entity-specific targets
    '''
    # ([bs *] # facts, seqlen): [facts about entity 0, facts about entity 1, etc.]
    batch_all_facts = {}  # fill in entities
    token_all_facts = {}
    all_vectors = {}
    fact_to_idx = {}
    # all permutations (orderings)...
    ent_pairs = itertools.combinations([None, *entity_name_to_types], 2)
    if len(entity_name_to_types) > 50: ent_pairs = tqdm(ent_pairs)
    for ent_pair in ent_pairs:
        assert not "I" in ent_pair and not "P" in ent_pair
        # if ent_pair[1] is not None and "-= " in ent_pair[1]: ent_pair[1] = ent_pair[1][3:-3].lower()
        entset = EntitySet(ent_pair)
        entset_serialize = EntitySet.serialize(entset)
        if entset_serialize not in batch_all_facts: batch_all_facts[entset_serialize] = []
        # set invalid facts to `invalid`, to indicate to model not to connect to those
        # (can't delete them straightforwardly as otherwise labels would be unaligned...)
        for fact in all_facts:
            if 'entity2' in fact: continue
            # TODO
            if entset.has_none:
                # for facts with 2 args, ent_pair cannot have 1 arg
                if 'entity0' in fact and 'entity1' in fact: continue
                # fill whichever is present
                fact_filled = fact.replace('entity0', entset.nonNone_ent).replace('entity1', entset.nonNone_ent)
            else:
                # for facts with 1 arg, ent_pair cannot have 2 args
                if 'entity0' not in fact or 'entity1' not in fact: continue
                # both present
                fact_filled = fact.replace('entity0', entset[0]).replace('entity1', entset[1])
                fact_filled_reverse = fact.replace('entity0', entset[1]).replace('entity1', entset[0])
                batch_all_facts[entset_serialize].append(fact_filled_reverse)
            batch_all_facts[entset_serialize].append(fact_filled)
        fact_to_idx[entset_serialize] = {fact: idx for idx, fact in enumerate(batch_all_facts[entset_serialize])}
        # input_ids: ([bs *] # facts, seqlen), attention_mask: ([bs *] # facts, seqlen)
        token_all_facts[entset_serialize] = tokenizer(batch_all_facts[entset_serialize], return_tensors='pt', padding=True, truncation=True)
        
        tokens = token_all_facts[entset_serialize].to(device)
        vectors = state_encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], return_dict=True).last_hidden_state.to('cpu')

        '''
        model forward
        '''
        # encode everything
        # (bs * # facts, seqlen, embeddim)
        all_vectors[entset_serialize] = {
            'input_ids': vectors,
            'attention_mask': token_all_facts[entset_serialize]['attention_mask'],
        }
    probe_outs['state_to_idx'] = fact_to_idx
    probe_outs['idx_to_state'] = batch_all_facts
    probe_outs['all_entity_vectors'] = all_vectors
    probe_outs['all_entity_inputs'] = token_all_facts

    return probe_outs


ENTITIES_SIMPLE = ['player', 'inventory', 'wooden door', 'chest drawer', 'antique trunk', 'king-size bed', 'old key', 'lettuce', 'tomato plant', 'milk', 'shovel', 'toilet', 'bath', 'sink', 'soap bar', 'toothbrush', 'screen door', 'set of chairs', 'bbq', 'patio table', 'couch', 'low table', 'tv', 'half of a bag of chips', 'remote', 'refrigerator', 'counter', 'stove', 'kitchen island', 'bell pepper', 'apple', 'note']
ROOMS_SIMPLE = ['garden', 'bathroom', 'kitchen', 'bedroom', 'backyard', 'living room']
control_pairs_simple = [
    ('player', 'inventory'), ('inventory', 'player'), ('wooden door', 'screen door'), ('screen door', 'refrigerator'), ('refrigerator', 'counter'), ('counter', 'stove'),
    ('stove', 'kitchen island'), ('kitchen island', 'apple'), ('apple', 'note'), ('note', 'tomato plant'), ('tomato plant', 'wooden door'), ('bell pepper', 'milk'), ('milk', 'shovel'),
    ('shovel', 'half of a bag of chips'), ('half of a bag of chips', 'bell pepper'), ('toilet', 'bath'), ('bath', 'sink'), ('sink', 'soap bar'), ('soap bar', 'toothbrush'),
    ('toothbrush', 'toilet'), ('lettuce', 'couch'), ('couch', 'low table'), ('low table', 'tv'), ('tv', 'remote'), ('remote', 'lettuce'), ('chest drawer', 'antique trunk'),
    ('antique trunk', 'king-size bed'), ('king-size bed', 'old key'), ('old key', 'chest drawer'), ('set of chairs', 'bbq'), ('bbq', 'patio table'), ('patio table', 'set of chairs')
]
control_pairs_with_rooms_simple = control_pairs_simple + [('garden', 'bathroom'), ('bathroom', 'kitchen'), ('kitchen', 'bedroom'), ('bedroom', 'backyard'), ('backyard', 'living room'), ('living room', 'garden')]

control_tgt_to_mention_simple = {pair[0]: pair[1] for pair in control_pairs_simple}
control_tgt_to_mention_with_rooms_simple = {pair[0]: pair[1] for pair in control_pairs_with_rooms_simple}
control_mention_to_tgt_simple = {pair[1]: pair[0] for pair in control_pairs_simple}
control_mention_to_tgt_with_rooms_simple = {pair[1]: pair[0] for pair in control_pairs_with_rooms_simple}
