import torch
import os
import glob
import json
from tqdm import tqdm
from data.textworld.tw_dataloader import TWDataset, load_possible_pairs
from data.textworld.parse_tw import parse_facts_to_nl, parse_nl_to_facts
from data.textworld.utils import ENTITIES_SIMPLE, ROOMS_SIMPLE, EntitySet, get_relevant_facts_about
from data.alchemy.utils import translate_nl_to_states

from transformers import BartTokenizerFast, T5TokenizerFast
import textworld
import warnings
import itertools

import argparse


def get_prf1(relevant_full_true_facts, pred_true_facts, pred_false_facts, gt_true_facts, gt_false_facts):
    # true recall/precision, false recall/precision
    # T precision
    num_true_all_fact_overlap = len(set(relevant_full_true_facts).intersection(set(pred_true_facts)))
    num_true_pred = len(set(pred_true_facts))
    true_precision = num_true_all_fact_overlap / num_true_pred if num_true_pred > 0 else 1.0
    num_true_gt_fact_overlap = len(set(gt_true_facts).intersection(set(pred_true_facts)))
    num_true_gt = len(set(gt_true_facts))
    true_recall = num_true_gt_fact_overlap / num_true_gt if num_true_gt > 0 else 1.0
    true_f1 = 2 * true_precision * true_recall / (true_precision + true_recall) if true_precision + true_recall > 0 else 0
    
    # F precision
    # alltrue' (intersect) predfalse = predfalse - alltrue
    num_false_all_fact_overlap = len(set(pred_false_facts) - set(relevant_full_true_facts))
    num_false_pred = len(set(pred_false_facts))
    false_precision = num_false_all_fact_overlap / num_false_pred if num_false_pred > 0 else 1.0
    num_false_gt_fact_overlap = len(set(gt_false_facts).intersection(set(pred_false_facts)))
    num_false_gt = len(set(gt_false_facts))
    false_recall = num_false_gt_fact_overlap / num_false_gt if num_false_gt > 0 else 1.0
    false_f1 = 2 * false_precision * false_recall / (false_precision + false_recall) if false_precision + false_recall > 0 else 0
    return [true_precision, true_recall, true_f1], [false_precision, false_recall, false_f1]


def split_relational_property_facts(all_facts, game_id, game_state=None, inform7_game=None, cached_templates=None):
    if len(all_facts) == 0:
        return all_facts, all_facts, cached_templates
    if cached_templates is None: cached_templates = {}
    property_facts = []
    rel_facts = []
    non_wellformed_facts = []
    for f, fact in enumerate(all_facts):
        is_property = False
        is_relation = False
        if 'is closed' in fact or 'is open' in fact or 'is locked' in fact or 'is nowhere' in fact or 'is edible' in fact:
            property_facts.append(fact)
            is_property = True
        if 'is in ' in fact or 'is on ' in fact or 'is at ' in fact or 'The player carries ' in fact or 'The matching key' in fact or 'is mapped ' in fact:
            rel_facts.append(fact)
            is_relation = True
        assert not (is_property and is_relation)
        if not is_property and not is_relation:
            parsed_fact, cached_templates = parse_nl_to_facts([fact], game_state, game_id, cached_templates=cached_templates, inform7_game=inform7_game)
            if len(parsed_fact) == 0:
                non_wellformed_facts.append(fact)
            else:
                assert len(parsed_fact) == 1
                parsed_fact = parsed_fact[0]
                if len(parsed_fact['arguments']) == 2: rel_facts.append(fact)
                elif len(parsed_fact['arguments']) == 1: property_facts.append(fact)
                else: non_wellformed_facts.append(fact)
    return rel_facts, property_facts, non_wellformed_facts, cached_templates


def get_rel_prop_prf1(
    full_gt_state, results, game_id, game_state=None, inform7_game=None, cached_templates={},
):
    # distinguish relational and property ems
    rel_full_gt_state, prop_full_gt_state, non_wellformed_gt_state, cached_templates = split_relational_property_facts(
        full_gt_state, game_id, game_state=game_state, inform7_game=inform7_game, cached_templates=cached_templates)
    assert len(non_wellformed_gt_state) == 0
    prop_states = {}
    rel_states = {}
    for key in ['queried_gen_state', 'queried_gt_state']:
        prop_states[key] = {}
        rel_states[key] = {}
        for tf in ['true', 'false']:
            rel_state, prop_state, non_wellformed_state, cached_templates = split_relational_property_facts(
                results[key][tf], line['game_id'], game_state=game_state, inform7_game=inform7_game, cached_templates=cached_templates)
            prop_states[key][tf] = prop_state
            rel_states[key][tf] = rel_state
    rel_queried_true_metrics, rel_queried_false_metrics = get_prf1(
        rel_full_gt_state, rel_states['queried_gen_state']['true'], rel_states['queried_gen_state']['false'],
        rel_states['queried_gt_state']['true'], rel_states['queried_gt_state']['false'],
    )
    prop_queried_true_metrics, prop_queried_false_metrics = get_prf1(
        prop_full_gt_state, prop_states['queried_gen_state']['true'], prop_states['queried_gen_state']['false'],
        prop_states['queried_gt_state']['true'], prop_states['queried_gt_state']['false'],
    )
    return rel_queried_true_metrics, rel_queried_false_metrics, prop_queried_true_metrics, prop_queried_false_metrics



parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='bart', choices=['t5', 'bart'])
parser.add_argument('--domain', type=str, required=True, choices=['alchemy', 'textworld'])
parser.add_argument('--pred_files', type=str, help="comma-separated `.jsonl` file of model outputs")
parser.add_argument('--use_remap_domain', action='store_true', default=False, help="evaluate on remap domain")
parser.add_argument('--remap_fn', type=str, default=None, help="file path to remap probe outputs (required if `--use_remap_domain`)")
parser.add_argument('--single_side_probe', action='store_true', default=False, help="`pred_files` include 1-side probe outputs")
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()

arch = args.arch
eval_fns = args.pred_files.split(',')

if args.domain == 'alchemy':
    NUM_EXAMPLES = 980
    NUM_BEAKERS = 7
    for fn in eval_fns:
        EM_perquery = 0
        num_lines = 0
        examples = []
        # examples = {}
        ex_num = 0
        with open(fn) as f:
            for line in f:
                num_lines += 1
                line = json.loads(line)
                bn = int(line['gold_state'].split(':')[0])
                EM_perquery += line['gen_state'] == line['gold_state']
                if bn > 1:
                    examples[ex_num]['pred'] += f' {line["gen_state"]}'
                    examples[ex_num]['gold'] += f' {line["gold_state"]}'
                    examples[ex_num]['n_beakers_correct'] += int(line['gen_state'] == line['gold_state'])
                    ex_num = (ex_num + 1) % NUM_EXAMPLES
                else:
                    prior = line['prior'].split('<s>')[-1].split('</s>')[0]
                    init_ctxt = translate_nl_to_states(prior.split('. ')[0], 'alchemy')
                    examples.append({'prior': f'{init_ctxt}. {prior.split(". ")[-1]}', 'pred': line['gen_state'], 'gold': line['gold_state'], 'n_beakers_correct': int(line['gen_state'] == line['gold_state'])})
        print(fn)
        print(f'Entity Exact Match: {EM_perquery / num_lines}')

        EM = 0
        n_bkrmatch = 0
        for ex in examples:
            EM += ex['pred'] == ex['gold']
            gold_bkrs = ex['gold'].split(' ')
            pred_bkrs = ex['pred'].split(' ')
            assert len(gold_bkrs) == len(pred_bkrs) == NUM_BEAKERS
            ex_num_bkrmatch = 0
            for bkr in range(NUM_BEAKERS): ex_num_bkrmatch += gold_bkrs[bkr] == pred_bkrs[bkr]
            ex_num_bkrmatch /= NUM_BEAKERS
            n_bkrmatch += ex_num_bkrmatch

        print(f'State Exact Match: {EM / len(examples)}')

if args.domain == 'textworld':
    data_type = 'simple'
    local_files_only = True
    split_rel_prop = True

    if args.single_side_probe:
        # eval_fns = [single_fn]
        entities = ENTITIES_SIMPLE + ROOMS_SIMPLE
        all_entities_list = list(itertools.combinations([None, *entities], 2))
        possible_pairs = load_possible_pairs(pair_out_file='tw_data/training_traces_tw-simple/entity_pairs.json')

    print(eval_fns)
    eval_fns = [eval_fn for eval_fn in eval_fns if os.path.exists(eval_fn)]

    if args.use_remap_domain:
        ctxt_to_remap_entities = {}
        with open(args.remap_fn) as f:
            for line in f:
                line = json.loads(line)
                line['prev_context'] = line['prev_context'].strip()
                if line['prev_context'] not in ctxt_to_remap_entities:
                    ctxt_to_remap_entities[line['prev_context']] = []
                ctxt_to_remap_entities[line['prev_context']].append(line['entity'])

    for eval_fn in eval_fns:
        control_input = '.control_inp' in eval_fn
        errors = {}
        all_fact_counts = {}
        print(eval_fn)
        if arch == 'bart':
            tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base', local_files_only=local_files_only)
        elif arch == 't5':
            tokenizer = T5TokenizerFast.from_pretrained('t5-base', local_files_only=local_files_only)
        else:
            raise NotImplementedError()
        gamefile = f'tw_data/{data_type}_games'
        for fn in glob.glob(os.path.join(gamefile, 'train/*.ulx')):
            env = textworld.start(fn)
            game_state = env.reset()
            game_kb = game_state['game'].kb.inform7_predicates
            inform7_game = env._inform7
            break

        # get gts
        dataset = TWDataset(f'tw_data/{data_type}_traces', tokenizer, 'dev', max_seq_len=512, max_data_size=500)
        ctxt_to_gt_state = {}
        for entry in dataset:
            if not control_input:
                ctxt = tokenizer.decode(tokenizer.encode(entry['contexts']), skip_special_tokens=True).strip()
            else:
                ctxt = entry['contexts'].strip()
            ctxt_to_gt_state[ctxt] = {subkey: {tf: parse_facts_to_nl(entry['final_states'][subkey][tf], inform7_game) for tf in ['true', 'false']} for subkey in entry['final_states'] if 'belief_facts' in subkey}
            ctxt_to_gt_state[ctxt]['full_facts'] = parse_facts_to_nl(entry['final_states']['full_facts'], inform7_game)

        # get preds
        # by context
        results = {}
        queried_ent_em = [0.,0]
        cached_templates = {}
        with open(eval_fn) as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            line['prev_context'] = line['prev_context'].strip()
            if not args.use_remap_domain or line['entity'] in ctxt_to_remap_entities[line['prev_context']]:
                if line['prev_context'] not in results:
                    assert line['prev_context'] in ctxt_to_gt_state
                    results[line['prev_context']] = {
                        'queried_gt_state': {'true': [], 'false': []}, 'queried_gen_state': {'true': [], 'false': []}, 'queried_entities': [],
                        'gt_state': {'true': ctxt_to_gt_state[line['prev_context']]['belief_facts']['true'], 'false': ctxt_to_gt_state[line['prev_context']]['belief_facts']['false']},
                        'full_gt_state': ctxt_to_gt_state[line['prev_context']]['full_facts'],
                    }
                # create all entity pairs
                if args.single_side_probe:
                    queried_gen_state = {'true': [], 'false': []}
                    queried_gt_state = {'true': [], 'false': []}
                    for tf in ['true', 'false']:
                        gt_facts, cached_templates = parse_nl_to_facts(line['gt_state'][tf], game_state, line['game_id'], cached_templates=cached_templates, inform7_game=inform7_game)
                        gen_facts, cached_templates = parse_nl_to_facts(line['gen_state'][tf], game_state, line['game_id'], cached_templates=cached_templates, inform7_game=inform7_game)
                        line['gt_state'][tf] = gt_facts
                        line['gen_state'][tf] = gen_facts
                    for ent_list in all_entities_list:
                        entset = EntitySet(ent_list)
                        if not entset.has_nonNone or entset not in possible_pairs[line['game_id']]: continue
                        for tf in ['true', 'false']:
                            queried_gt_state[tf] += parse_facts_to_nl(get_relevant_facts_about(entset, line['gt_state'][tf], None, None, exact_arg_count=True, exact_arg_order=False), inform7_game)
                            queried_gen_state[tf] += parse_facts_to_nl(get_relevant_facts_about(entset, line['gen_state'][tf], None, None, exact_arg_count=True, exact_arg_order=False), inform7_game)
                    for tf in ['true', 'false']:
                        line['gt_state'][tf] = queried_gt_state[tf]
                        line['gen_state'][tf] = queried_gen_state[tf]
                for tf in ['true', 'false']:
                    results[line['prev_context']]['queried_gt_state'][tf] += line['gt_state'][tf]
                    results[line['prev_context']]['queried_gen_state'][tf] += line['gen_state'][tf]
                results[line['prev_context']]['queried_entities'].append(line['entity'])
                results[line['prev_context']]['game_id'] = line['game_id']
                ent_em = (line['gt_state']['true'] == line['gen_state']['true']) and (line['gt_state']['false'] == line['gen_state']['false'])

                entity = json.loads(line['entity'])
                queried_ent_em[0] += ent_em
                queried_ent_em[1] += 1
        print(len(dataset))
        # assert len(dataset) == len(results)

        all_queried_true_metrics = torch.tensor([0.,0.,0.])
        all_queried_false_metrics = torch.tensor([0.,0.,0.])
        all_full_true_metrics = torch.tensor([0.,0.,0.])
        all_full_false_metrics = torch.tensor([0.,0.,0.])
        queried_em = 0
        full_em = 0
        relational_q_em = 0
        property_q_em = 0
        for context in results:
            if split_rel_prop:
                rel_queried_true_metrics, rel_queried_false_metrics, prop_queried_true_metrics, prop_queried_false_metrics = get_rel_prop_prf1(
                    results[context]['full_gt_state'], results[context], results[context]['game_id'],
                    game_state=game_state, inform7_game=inform7_game, cached_templates=cached_templates,
                )
                relational_q_em += (rel_queried_true_metrics[-1] == 1) and (rel_queried_false_metrics[-1] == 1)
                property_q_em += (prop_queried_true_metrics[-1] == 1) and (prop_queried_false_metrics[-1] == 1)
            # get overall ems
            queried_true_metrics, queried_false_metrics = get_prf1(
                results[context]['full_gt_state'], results[context]['queried_gen_state']['true'], results[context]['queried_gen_state']['false'],
                results[context]['queried_gt_state']['true'], results[context]['queried_gt_state']['false'],
            )
            queried_em += (queried_true_metrics[-1] == 1) and (queried_false_metrics[-1] == 1)
            true_diff, false_diff = set(), set()
            if queried_true_metrics[-1] != 1:
                true_diff = set(results[context]['queried_gen_state']['true']).symmetric_difference(set(results[context]['queried_gt_state']['true']))
            if queried_false_metrics[-1] != 1:
                false_diff = set(results[context]['queried_gen_state']['false']).symmetric_difference(set(results[context]['queried_gt_state']['false']))
            diffs = true_diff.union(false_diff)
            if len(diffs) > 0:
                for diff in diffs:
                    if 'The matching key' in diff:
                        continue
                    if diff not in errors: errors[diff] = 0
                    errors[diff] += 1
                # print(context)
                # print({'true': true_diff, 'false': false_diff})
            for state_key in ['queried_gen_state', 'queried_gt_state']:
                for tf in ['true', 'false']:
                    for fact in results[context][state_key][tf]:
                        if 'The matching key' in fact:
                            continue
                        if fact not in all_fact_counts: all_fact_counts[fact] = 0
                        all_fact_counts[fact] += 1
            all_queried_true_metrics += torch.tensor(queried_true_metrics)
            all_queried_false_metrics += torch.tensor(queried_false_metrics)
            full_true_metrics, full_false_metrics = get_prf1(
                results[context]['full_gt_state'], results[context]['queried_gen_state']['true'], results[context]['queried_gen_state']['false'],
                results[context]['gt_state']['true'], results[context]['gt_state']['false'],
            )
            full_em += (full_true_metrics[-1] == 1) and (full_false_metrics[-1] == 1)
            all_full_true_metrics += torch.tensor(full_true_metrics)
            all_full_false_metrics += torch.tensor(full_false_metrics)

        all_queried_true_metrics /= len(results)
        all_queried_false_metrics /= len(results)
        all_full_false_metrics /= len(results)
        all_full_true_metrics /= len(results)
        queried_em /= len(results)
        full_em /= len(results)
        relational_q_em /= len(results)
        property_q_em /= len(results)

        print(f'All queried results: State EM = {queried_em}')
        if split_rel_prop:
            print(f'        Relations - {relational_q_em}')
            print(f'        Properties - {property_q_em}')
        print(f'    true - {all_queried_true_metrics.tolist()}')
        print(f'    false - {all_queried_false_metrics.tolist()}')
        print(f'    Entity EM - {queried_ent_em[0]/queried_ent_em[1]}')
        if args.verbose:
            print(errors)
            print(all_fact_counts)

        print(f'All full results: State EM = {full_em}')
        print(f'    true - {all_full_true_metrics.tolist()}')
        print(f'    false - {all_full_false_metrics.tolist()}')
