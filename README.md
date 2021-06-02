# Implicit Representations of Meaning in Neural Language Models
## Preliminaries
Create and set up a conda environment as follows:
```bash
conda create -n state-probes python=3.7
conda activate state-probes
pip install -r requirements.txt
```

Before running any command below, run
```bash
export PYTHONPATH=.
```


## Data
The Alchemy data is downloaded from their website.
```bash
wget https://nlp.stanford.edu/projects/scone/scone.zip
unzip scone.zip
```
The synthetic version of alchemy was generated by running:
```bash
echo 0 > id #the code requires a file called id with a number in it ...
python alchemy_artificial_generator.py --num_scenarios 3600 --output synth_alchemy_train
python alchemy_artificial_generator.py --num_scenarios 500 --output synth_alchemy_dev
python alchemy_artificial_generator.py --num_scenarios 900 --output synth_alchemy_test
```
You can also just download our generated data through:
```bash
wget http://web.mit.edu/bzl/www/synth_alchemy.tar.gz
tar -xzvf synth_alchemy.tar.gz
```

The Textworld data is under
```bash
wget http://web.mit.edu/bzl/www/tw_data.tar.gz
tar -xzvf tw_data.tar.gz
```


## LM Training
To train a BART or T5 model on Alchemy data
```bash
python scripts/train_alchemy.py [--synthetic] --encode_init_state NL [--no_pretrain] --arch [t5|bart] --save_path sconeModels/[real|synth][_nopre][t5|bart]_encInitState=NL.p [--local_files_only]
```
Saves under `sconeModels/*`

To train a BART or T5 model on Textworld data
```bash
python scripts/train_textworld.py --data tw_data/simple_traces --gamefile tw_games/simple_games [--no_pretrain] --arch [t5/bart] [--local_files_only]
```
Saves under `twModels/*`


## Probe Training & Evaluation
### Alchemy
The main probe command is as follows:
```bash
python probe_alchemy.py \
    --arch [bart|t5] --lm_save_path <path_to_lm_checkpoint> [--no_pretrain]
    --encode_init_state NL --nonsynthetic \
    --probe_target single_beaker_final.NL --localizer_type single_beaker_init_full \
    --probe_type linear --probe_agg_method avg \
    --encode_tgt_state NL.[bart|t5] --tgt_agg_method avg \
    --batchsize 128 --eval_batchsize 1024 --lr 1e-4
```
Add `--control_input` for No LM experiment. Change `--probe_target` to `single_beaker_init.NL` to decode initial state.

For evaluation, add `--eval_only --probe_save_path <path_to_probe_checkpoint>`

For localization experiments, set `--localizer_type single_beaker_init_{$i}.offset{$off}` for some token `i` in `{article, pos.[R0|R1|R2], beaker.[R0|R1], verb, amount, color, end_punct}` and some integer offset `off` between 0 and 6.

Intervention experiments results followed from running the script:
```bash
python scripts/intervention.py --arch [bart|t5] --encode_init_state NL --create_type drain_1 --lm_save_path <path_to_lm_checkpoint> [--get_kl]
```
which creates two contexts

### Textworld
Begin by creating the full set of encoded proposition representations 
```bash
python scripts/get_all_tw_facts.py --state_model_arch t5 --probe_target belief_facts_pair --state_model_path ../state-probes-TW/model_checkpoints/nonpre_noft_t5.p --out_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_nonpre_noft_t5.p --local_files_only
```

Run the probe with
```bash
python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/nonpre_noft_t5.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_nonpre_noft_t5.p --probe_save_path probe_models/nopre_noft_t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```

T5, probe -pretrain, +finetune [PROBING]
```
cp model_checkpoints/nonpre_noft_t5.p model_checkpoints/nonpre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p

python train_lm.py --data tw_games/training_traces_tw-simple --gamefile tw_games/training_tw-simple --arch t5 --no_pretrain --local_files_only --data_type textworld --epochs 100

python scripts/get_all_tw_facts.py --state_model_arch t5 --probe_target belief_facts_pair --state_model_path ../state-probes-TW/model_checkpoints/nonpre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --out_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_nonpre_t5.p --local_files_only

sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/nonpre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_nonpre_t5.p --probe_save_path probe_models/nopre_t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only

for epoch in 0 5 10 15 20 30 35
do
sbatch scripts/temp_exp_wrapper.sh $epoch
done
```

T5, probe +pretrain, -finetune [DONE]
```
python scripts/get_all_tw_facts.py --state_model_arch t5 --probe_target belief_facts_pair --state_model_path pretrain --out_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_noft_t5.p --local_files_only

sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch bart --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.bart --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_noft_bart.p --probe_save_path probe_models/noft_bart_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```

T5, probe +pretrain, +finetune [PROBING]
```
python train_lm.py --data tw_games/training_traces_tw-simple --gamefile tw_games/training_tw-simple --arch t5 --local_files_only --data_type textworld --epochs 100

python scripts/get_all_tw_facts.py --state_model_arch t5 --probe_target belief_facts_pair --state_model_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --out_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --local_files_only

sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch bart --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.bart --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_bart_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_bart.p --probe_save_path probe_models/bart_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```

T5, probe +pretrain, +finetune, localizer first
```
sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_first --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_first_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```

T5, probe +pretrain, +finetune, localizer last
```
sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_last --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_last_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```

No LM [PROBING]
```
sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all.control_inp_avgavg_final.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only --control_input
```

No change
```
# train
sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target init.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all.avgavg_init.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only

# eval
sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all.avgavg_init.full_belief_facts_pair_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only --eval_only
```

Remap
```
sbatch scripts/slurm_wrapper.sh python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_pair.control_with_rooms --localizer_type actions.belief_facts_pair_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_pair_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_pair_all_avgavg_final.full_belief_facts_pair.control_with_rooms_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```

Single-side
```
python scripts/get_all_tw_facts.py --state_model_arch bart --probe_target belief_facts_single --state_model_path ../state-probes-TW/model_checkpoints/pre_bart_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --out_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_single_bart.p --local_files_only

python probe_textworld.py --arch t5 --data ../state-probes-TW/tw_games/training_traces_tw-simple --gamefile ../state-probes-TW/tw_games/training_tw-simple --probe_target final.full_belief_facts_single --localizer_type actions.belief_facts_single_all --encode_tgt_state NL.t5 --probe_type 3linear_classify --probe_agg_method avg --tgt_agg_method avg --max_seq_len 512 --lm_save_path ../state-probes-TW/model_checkpoints/pre_t5_lr1e-05_training_traces_tw-simple_4000_seed42/lang_models/best_lm_loss.p --ents_to_states_file ../state-probes-TW/tw_games/training_traces_tw-simple/entities_to_facts/belief_facts_single_t5.p --probe_save_path probe_models/t5_same_state_lr1e-05_training_traces_tw-simple_4000_seed42/enctgt_3linear_classify_actions.belief_facts_single_all_avgavg_final.full_belief_facts_single_4000_seed42.p --eval_batchsize 256 --batchsize 32 --local_files_only
```