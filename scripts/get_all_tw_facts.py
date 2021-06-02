import os
import textworld
import json
import sys
from data.textworld.tw_dataloader import TWDataset
import argparse
from transformers import (
    BartConfig, BartTokenizerFast, BartForConditionalGeneration,
    T5Config, T5TokenizerFast, T5ForConditionalGeneration,
)
from tqdm import tqdm
import itertools
import torch
from data.textworld.utils import EntitySet, gen_possible_pairs, gen_all_facts


def main(data_dir, gamefile, out_file, state_model_path, state_model_arch, probe_target, local_files_only, state_model_layers=None, device='cuda'):
    if state_model_arch == 'bart':
        model_class = BartForConditionalGeneration
        config_class = BartConfig
        model_fp = 'facebook/bart-base'
        tokenizer = BartTokenizerFast.from_pretrained(model_fp, local_files_only=local_files_only)
    elif state_model_arch == 't5':
        model_class = T5ForConditionalGeneration
        config_class = T5Config
        model_fp = 't5-base'
        tokenizer = T5TokenizerFast.from_pretrained(model_fp, local_files_only=local_files_only)
    else:
        raise NotImplementedError()

    if state_model_path and state_model_path == 'pretrain':
        state_model = model_class.from_pretrained(model_fp, local_files_only=local_files_only)
    else:
        config = config_class.from_pretrained(model_fp, local_files_only=local_files_only)
        if state_model_layers is not None:
            if state_model_arch == 'bart':
                setattr(config, 'num_hidden_layers', state_model_layers)
                setattr(config, 'encoder_layers', state_model_layers)
                setattr(config, 'decoder_layers', state_model_layers)
            elif state_model_arch == 't5':
                setattr(config, 'num_layers', state_model_layers)
                setattr(config, 'num_decoder_layers', state_model_layers)
        state_model = model_class(config)
        if state_model_path:
            state_model.load_state_dict(torch.load(state_model_path, map_location=torch.device('cpu')))
            state_model_path = os.path.split(state_model_path)[-1].replace('.p', '')
    state_model.to(device)
    state_model.eval()
    state_encoder = state_model.get_encoder()
    
    if out_file is None:
        out_dir = os.path.join(data_dir, 'entities_to_facts')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f'{probe_target}_{state_model_arch}_state_model_{state_model_path}.p'
        )
    print(f"Saving state vectors to {out_file}")

    dev_dataset = TWDataset(data_dir, tokenizer, 'dev', max_seq_len=float("inf"), max_data_size=float("inf"))
    print(f"Loaded dev data: {len(dev_dataset)} examples")
    dataset = TWDataset(data_dir, tokenizer, 'train', max_seq_len=float("inf"), max_data_size=float("inf"))
    print(f"Loaded train data: {len(dataset)} examples")
    game_ids = dev_dataset.get_gameids() + dataset.get_gameids()
    if 'pair' in probe_target: ent_set_size=2
    if 'single' in probe_target: ent_set_size=1
    with torch.no_grad():
        probe_outs = gen_all_facts(gamefile, state_encoder, None, tokenizer, tqdm(game_ids), ent_set_size, device)
    torch.save(probe_outs, out_file)

    if 'pair' in probe_target:
        pair_out_file = os.path.join(data_dir, 'entity_pairs.json')
        if not os.path.exists(pair_out_file):
            print(f"Computing entity pairs to {pair_out_file}")
            possible_pairs, type_to_gid_to_ents = gen_possible_pairs(gamefile, tqdm(game_ids))
            possible_pairs_serialized = {}
            for gameid in possible_pairs:
                possible_pairs_serialized[gameid] = [EntitySet.serialize(pair) for pair in possible_pairs[gameid]]
            # serialize
            json.dump(possible_pairs_serialized, open(pair_out_file, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='tw_data/simple_traces')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gamefile', type=str, default='tw_data/simple_games')
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--state_model_path', type=str, default=None, help='None, `pretrain`, or filepath to checkpoint')
    parser.add_argument('--state_model_arch', type=str, choices=['t5', 'bart'])
    parser.add_argument('--probe_target', type=str, default='belief_facts_pair', choices=['belief_facts_pair', 'belief_facts_single'])
    parser.add_argument('--local_files_only', action='store_true', default=False)
    parser.add_argument('--override_num_layers', type=int, default=None)
    args = parser.parse_args()

    main(
        args.data_dir, args.gamefile, args.out_file, args.state_model_path,
        args.state_model_arch, args.probe_target, args.local_files_only,
        args.override_num_layers, args.device,
    )
