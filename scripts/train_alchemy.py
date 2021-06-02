import torch
from transformers import BartConfig, T5Config
from transformers import BartTokenizerFast, T5TokenizerFast
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from torch import nn
import numpy as np

from transformers import AdamW
from itertools import chain
import os
import random
from torch import optim
import argparse

from metrics.alchemy_metrics import check_val_consistency
from data.alchemy.parseScone import loadData, getBatches, getBatchesWithInit
from data.alchemy.scone_dataloader import convert_to_transformer_batches

NUM_POSITIONS = 10
POSITION_INDICES = range(NUM_POSITIONS)
COLORS = ['_', 'b', 'g', 'o', 'p', 'r', 'y']
COLORS_TO_INDEX = {
    color: index
    for (index, color) in enumerate(sorted(COLORS))
}
NUM_COLORS = len(COLORS)
STATE_ENC_DIM = NUM_POSITIONS * NUM_COLORS * 2


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='bart', choices=['t5', 'bart'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--eval_only', default=False, action='store_true')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--encode_init_state', type=str, default=False, choices=[False, 'raw', 'NL'])
parser.add_argument('--seed', type=int, default=45)
parser.add_argument('--no_context', action='store_true')
parser.add_argument('--synthetic', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no_pretrain', action='store_true', default=False)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--local_files_only', action='store_true', default=False)
args = parser.parse_args()

pretrained = not args.no_pretrain
# make save path
os.makedirs('sconeModels', exist_ok=True)
if not args.save_path:
    savePath = f'sconeModels/{"synth" if args.synthetic else "real"}_{"pre" if pretrained else "nopre"}{args.arch}_encInitState={args.encode_init_state}.p'
else: savePath = args.save_path
random.seed(args.seed)

# creating model
load_model = False
if os.path.exists(savePath):
    model_dict = torch.load(savePath)
    load_model = True

if args.arch == 'bart':
    model_class = BartForConditionalGeneration
    config_class = BartConfig
    model_fp = 'facebook/bart-base'
    tokenizer = BartTokenizerFast.from_pretrained(model_fp, local_files_only=args.local_files_only)
elif args.arch == 't5':
    model_class = T5ForConditionalGeneration
    config_class = T5Config
    model_fp = 't5-base'
    tokenizer = T5TokenizerFast.from_pretrained(model_fp, local_files_only=args.local_files_only)
else:
    raise NotImplementedError()

if not load_model: print("Creating LM model")
if not load_model and pretrained:
    model = model_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
else:
    config = config_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
    model = model_class(config)
if load_model:
    model.load_state_dict(model_dict)
    print("Loaded existing model checkpoint")
print(f"    model path: {savePath}")
model.to(args.device)
optimizer = AdamW(list(model.parameters()), lr=args.lr)

# loading data
dataset, lang_v, state_v = loadData(split="train", kind="alchemy", synthetic=args.synthetic)
dev_dataset, lang_v_dev, state_v_dev = loadData(split="dev", kind="alchemy", synthetic=args.synthetic)
best_val_loss = 10**10
best_val_consistency = 0.0
best_epoch = -1
all_train_states = [" ".join(state) for _, _, state in [x for d in dataset for x in d.all_pairs()] ]
all_dev_states = [" ".join(state) for _, _, state in [x for d in dev_dataset for x in d.all_pairs()] ]
random.shuffle(all_dev_states)

for i in range(args.epochs):
    if (i - best_epoch > args.patience) and (i - best_epoch > args.patience): break
    model.train()
    lang_train_losses = []

    for j, (inputs, lang_tgts, state_tgts, raw_state_targets, init_states) in enumerate(
        convert_to_transformer_batches(
            dataset, tokenizer, args.batchsize, random=random, include_init_state=args.encode_init_state, domain="alchemy", device=args.device,
        )
    ):
        optimizer.zero_grad()
        return_dict = model(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
            labels=lang_tgts['input_ids'], return_dict=True,
        )
        lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state
        encoder_outputs = (encoder_hidden,)

        lang_loss.backward()
        lang_train_losses.append(lang_loss)

        optimizer.step()
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, lang score: {lang_loss.item()}", flush=True)

    print(f"epoch {i}, average lang loss {sum(lang_train_losses).item()/len(lang_train_losses)}")
    model.eval()
    with torch.no_grad():
        tot_val_loss = 0
        n_val = 0
        n_val_consistent = 0
        for j, (inputs, lang_tgts, state_tgts, raw_state_targets, init_states) in enumerate(
            convert_to_transformer_batches(
                dev_dataset, tokenizer, args.batchsize, include_init_state=args.encode_init_state, domain="alchemy", device=args.device,
            )
        ):
            return_dict = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=lang_tgts['input_ids'], return_dict=True)
            lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state
            encoder_outputs = (encoder_hidden,)

            tot_val_loss += lang_loss*len(inputs['input_ids'])
            n_val += len(inputs['input_ids'])

            if args.synthetic:
                try:
                    n_val_consistent += check_val_consistency(
                        model, tokenizer, inputs, lang_tgts, init_states=init_states, included_init_state=args.encode_init_state,
                    )
                except:
                    n_val_consistent += check_val_consistency(
                        model, tokenizer, inputs, lang_tgts, init_states=init_states, included_init_state=args.encode_init_state,
                    )

        print("n_val", n_val)
        avg_val_loss = tot_val_loss.item()/n_val
        avg_val_consistency = n_val_consistent/n_val
        print(f"epoch {i}, avg val loss: {avg_val_loss}, fraction consistent: {avg_val_consistency}")
        
        if (args.synthetic and avg_val_consistency >= best_val_consistency) or (not args.synthetic and avg_val_loss <= best_val_loss):
            print("NEW BEST MODEL")
            model.epoch = i
            best_val_loss = avg_val_loss
            best_val_consistency = avg_val_consistency
            torch.save(model.state_dict(), savePath)
            best_epoch = i

        else:
            print(f"model val {'consistency' if args.synthetic else 'loss'} went {'down' if args.synthetic else 'up'}")
