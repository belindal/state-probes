import torch
from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader, Dataset, IterableDataset
import glob
from data.textworld.utils import (
    apply_mask_and_truncate, control_pairs_simple, control_pairs_with_rooms_simple,
    EntitySet, ENTITIES_SIMPLE, ROOMS_SIMPLE,
    gen_possible_pairs, get_relevant_facts_about,
    load_possible_pairs, load_negative_tgts, pad_stack, remap_entset,
    control_tgt_to_mention_simple, control_tgt_to_mention_with_rooms_simple,
)
from data.textworld.parse_tw import (
    parse_facts_to_nl,
)
import itertools
from transformers import PreTrainedTokenizerBase
import regex as re


class TWDataset(Dataset):
    def __init__(
        self, data_dir, tokenizer, data_split, max_seq_len=512,
        max_data_size=10000, interleave_state_in_ctxt=False, pred_action_and_response_joint=False,
        inform7_game=None, randseed=None, logger=None, *args, **kwargs,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_split = data_split
        self.max_seq_len = max_seq_len
        self.max_data_size = max_data_size
        self.interleave_state_in_ctxt = interleave_state_in_ctxt
        self.pred_action_and_response_joint = pred_action_and_response_joint
        self.inform7_game = inform7_game
        self.randseed = randseed
        self.logger = logger

        # build data
        self.load_data()
    
    def __len__(self):
        return len(self.data['contexts'])
    
    def __getitem__(self, i):
        item = {
            k: self.data[k][i] for k in self.data
        }
        item['data_idx'] = i
        return item
    
    def get_gameids(self):
        game_ids = []
        game_ids_set = set()
        # uniquify
        for fn in self.data['filenames']:
            game_id = fn.split('_')[0] 
            if game_id not in game_ids_set:
                game_ids.append(game_id)
                game_ids_set.add(game_id)
        return game_ids

    def load_data(self):
        init_actions_data = {'contexts': [], 'tgts': [], 'final_states': [], 'init_states': [], 'filenames': []}  # init state + actions
        n_states = 0
        files = sorted(glob.glob(os.path.join(os.path.join(self.data_dir, self.data_split), "*_states.txt")))
        if self.randseed:
            random.seed(self.randseed)
            random.shuffle(files)
        for fp in tqdm(files):
            all_actions = []  # actions that make up current file
            curr_action = []  # lines that make up current action
            n_cutoff_actions = 0  # num actions for max_seq_len (not strictly necessary, just ensures we don't run for too long)
            states = []
            # create all_actions (file, separated by commands, aka '>')
            langs_file = fp.replace('_states.txt', '.txt')
            with open(langs_file) as f:
                approx_num_toks = 0
                for line in f:
                    if (line.strip().startswith("***") and line.strip().endswith("***")) or approx_num_toks > self.max_seq_len:
                        # loop will always end on this condition, since "The End" is in all documents
                        break
                    line = line.strip() + ' | '
                    if line.startswith(">"):
                        action = ''.join(curr_action)
                        if approx_num_toks <= self.max_seq_len: n_cutoff_actions += 1
                        all_actions.append(action)
                        curr_action = []
                    curr_action.append(line)
                    approx_num_toks += len(self.tokenizer.tokenize(line))
                    if not self.pred_action_and_response_joint and line.startswith(">"):
                        # if action, add line immediately
                        action = ''.join(curr_action)
                        if approx_num_toks <= self.max_seq_len: n_cutoff_actions += 1
                        all_actions.append(action)
                        curr_action = []
                # get last part
                if line.startswith(">") and approx_num_toks + len(self.tokenizer.tokenize(line)) <= self.max_seq_len:
                    all_actions.append(''.join(curr_action))
                    if approx_num_toks + len(self.tokenizer.tokenize(line)) <= self.max_seq_len: n_cutoff_actions += 1
            # create final_states
            with open(fp) as f:
                num_lines = 0
                for line in f:
                    if num_lines > n_cutoff_actions + 1:  #+1 for initial state
                        break
                    state = json.loads(line)
                    states.append(state)
                    num_lines += 1

            if self.interleave_state_in_ctxt:
                all_actions = [
                    f"{all_actions[c]}[{'. '.join(parse_facts_to_nl(states[c]['added_belief_facts']['true'], self.inform7_game))}] ### "
                    for c in range(n_cutoff_actions)
                ]
                
            # create (context, next utterance, init_state, states) tuples for each dataset from all_actions
            # (all_actions[0], all_actions[1], states[0], states[0]);
            # (all_actions[0:1], all_actions[2], states[0], states[1]);
            # (all_actions[0:2], all_actions[3], states[0], states[2]);
            # ...
            # NOTE states[i] is state *after* `i`th action, so use (i-1) to get state immediately after context (actions 1...i-1)
            interacted_entities = set()
            s = 0  # after all_actions[0]
            for c in range(2,n_cutoff_actions):
                world = os.path.split(langs_file)
                world = os.path.join(os.path.split(world[0])[1], world[1])
                actions = ''.join(all_actions[1:c])
                tgt_action = all_actions[c].split('[')[0]
                increment_corresponding_state = all_actions[c-1].startswith(">")  # last action in context
                if increment_corresponding_state:
                    s += 1
                    n_states += 1

                goal = all_actions[0].split(' | ')[0]
                curr_context = ''.join([all_actions[0].replace(goal, ""), actions])
                init_actions_data['contexts'].append(curr_context)
                init_actions_data['tgts'].append(tgt_action)
                init_actions_data['init_states'].append(states[0])
                init_actions_data['final_states'].append(states[s])
                init_actions_data['filenames'].append(world)

                if len(init_actions_data['contexts']) >= self.max_data_size:
                    break
            if len(init_actions_data['contexts']) >= self.max_data_size:
                break
        for k in init_actions_data:
            assert len(init_actions_data[k]) == len(init_actions_data['contexts'])
        if self.logger: self.logger.info(f"Using files order: {init_actions_data['filenames']}")
        self.data = init_actions_data


class TWEntitySetDataset(TWDataset):
    """
    Same context for each entity pair, and return facts for each entity pair
    """
    def __init__(
        self, data_dir, tokenizer, data_split,
        ent_set_size, control, gamefile, state_key, tgt_state_key='final_states',
        max_seq_len=512, max_data_size=10000, interleave_state_in_ctxt=False,
        pred_action_and_response_joint=False, inform7_game=None, randseed=None, control_input=False,
        possible_pairs=None, precomputed_negs=None,
    ):
        """
        control_input: show same input
        """
        self.gamefile = gamefile
        self.inform7_game = inform7_game
        self.all_entities = ENTITIES_SIMPLE + ROOMS_SIMPLE
        self.control_tgt_to_mention = control_tgt_to_mention_simple
        self.control_tgt_to_mention_with_rooms = control_tgt_to_mention_with_rooms_simple
        self.all_rooms = ROOMS_SIMPLE
        self.tgt_state_key = tgt_state_key
        self.ent_set_size = ent_set_size
        self.control = control
        self.state_key = state_key

        self.possible_pairs = possible_pairs
        self.precomputed_negs = precomputed_negs
        super().__init__(
            data_dir=data_dir, tokenizer=tokenizer, data_split=data_split,
            max_seq_len=max_seq_len, max_data_size=max_data_size, 
            interleave_state_in_ctxt=interleave_state_in_ctxt, pred_action_and_response_joint=pred_action_and_response_joint,
            inform7_game=inform7_game, randseed=randseed,
        )

    def find_mention_in_ctxt(self, entity, ctxt):
        """
        check which form of entity is mentioned in the context
        If not found, returns `None`
        """
        if entity == "player": entity = "you"
        candidates = [f'[ |\n|\'|"][{entity[0].lower()}|{entity[0].upper()}]{entity[1:].lower()}[ |\n|,|\.|!|\?|\'|"]']
        candidates.append(f'-= {entity.title()} =-')
        all_mention_locations = None
        for cand in candidates:
            mention_location = list(re.finditer(cand, ctxt))
            if len(mention_location) > 0:
                if all_mention_locations is None: all_mention_locations = []
                all_mention_locations += mention_location
        return all_mention_locations

    def load_data(self):
        super().load_data()
        # compute negs here (if not already pre-loaded from file...)
        if self.precomputed_negs is None:
            if self.ent_set_size == 2:
                self.possible_pairs = load_possible_pairs(data_dir=self.gamefile, game_ids=self.get_gameids())
            self.precomputed_negs = load_negative_tgts(data_dir=self.gamefile, tokenizer=self.tokenizer, game_ids=self.get_gameids(), ent_set_size=self.ent_set_size)

        entities_data = {
            'contexts': [], 'tgts': [], 'init_states': [], 'tgt_states': [], 'filenames': [], 'game_ids': [],
            'entities': [], 'mentions': [], 'labels': [], 'all_states_tokenized': [], 'all_states_encoded': [],
        }
        print("Computing entities")
        # getting facts works only for returning 1 type of fact...
        for i in tqdm(range(len(self.data['contexts']))):
            context = self.data['contexts'][i]
            game_id = self.data['filenames'][i].split('_')[0]
            tgt = self.data['tgts'][i]
            init_state, tgt_state = self.data['init_states'][i][self.state_key], self.data[self.tgt_state_key][i][self.state_key]
            init_state = {tf: ' [SEP] '.join(parse_facts_to_nl(init_state[tf], self.inform7_game)) for tf in init_state}
            # get all entities mentioned in context
            # + transforms their names as appropriate
            entities = []
            ent2mentions = {}  # form of entity as mentioned in text
            for e in self.all_entities:
                if e == 'P': e = 'player'
                if e == 'I': e = 'inventory'
                e_in_ctxt = self.find_mention_in_ctxt(e, context)
                if not e_in_ctxt: continue
                ent2mentions[e] = e_in_ctxt
                if self.control:
                    ce = self.control_tgt_to_mention_with_rooms[e] if e in self.control_tgt_to_mention_with_rooms else e
                    ce_in_ctxt = self.find_mention_in_ctxt(ce, context)
                    if not ce_in_ctxt: continue
                    ent2mentions[ce] = ce_in_ctxt
                entities.append(e)

            all_entities_list = list(itertools.combinations([None, *entities], self.ent_set_size))
            # create all entity pairs
            for ent_list in all_entities_list:
                entset = EntitySet(ent_list)
                if not entset.has_nonNone or (self.possible_pairs is not None and entset not in self.possible_pairs[game_id]): continue

                # get all facts for list of entities
                ent_facts = {}
                for tf in tgt_state:
                    relevant_facts = get_relevant_facts_about(entset, tgt_state[tf], None, None, exact_arg_count=(self.ent_set_size > 1), exact_arg_order=False)
                    ent_facts[tf] = relevant_facts

                if self.control == 'control':
                    entset = EntitySet(remap_entset(entset, self.control_tgt_to_mention))
                if self.control == 'control_rooms':
                    entset = EntitySet(remap_entset(entset, self.control_tgt_to_mention_with_rooms))
                mentionset = remap_entset(entset, ent2mentions)

                entities_data['contexts'].append(context)
                entities_data['tgts'].append(tgt)
                entities_data['filenames'].append(self.data['filenames'][i])
                entities_data['game_ids'].append(game_id)
                entities_data['init_states'].append(init_state)
                ent_facts = {tf: ' [SEP] '.join(parse_facts_to_nl(ent_facts[tf], self.inform7_game)) for tf in ent_facts}
                entities_data['tgt_states'].append(ent_facts)
                entities_data['entities'].append(entset)
                entities_data['mentions'].append(mentionset)

                labels, all_states_inputs, all_states_vectors = self.get_matching_state_label(entset, ent_facts, self.precomputed_negs)
                entities_data['labels'].append(labels)
                entities_data['all_states_tokenized'].append(all_states_inputs)
                entities_data['all_states_encoded'].append(all_states_vectors)
        self.data = entities_data

    def get_matching_state_label(self, entset, target_state, precomputed_negs):
        '''
        create targets
        (pos/neg/unk)
        '''
        all_input_tokens = precomputed_negs['all_entity_inputs'][EntitySet.serialize(entset)].to('cpu')
        all_vectors = precomputed_negs['all_entity_vectors'][EntitySet.serialize(entset)]
        fact_to_idx = precomputed_negs['state_to_idx'][EntitySet.serialize(entset)]
        all_inputs = precomputed_negs['idx_to_state'][EntitySet.serialize(entset)]

        '''
        create labels
        '''
        labels = [0 for _ in range(len(fact_to_idx))]
        for i, tf in enumerate(target_state):
            if len(target_state[tf]) > 0:
                for fact in target_state[tf].split(' [SEP] '):
                    if 'carries' in fact:
                        if fact not in fact_to_idx:
                            fact = fact.replace("The player carries ", "") + " is in the inventory"
                    fact = f'{fact[0].upper()}{fact[1:]}'
                    labels[fact_to_idx[fact]] = i + 1
        return labels, all_input_tokens, all_vectors


class TWEntitySetDataLoader(DataLoader):
    def __init__(
        self, dataset: TWDataset, tokenizer: PreTrainedTokenizerBase, batch_size: int, control_input: bool = False, device: str = 'cuda',
    ):
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        self.control_input = control_input
        self.device = device
        
    """
    state_keys_to_get: [(init/final_state, key); (init/final_state, key)]
    nnegs: # negatives to get (set 0 to not get negatives, inf to get all negatives)
    npos: # positives to get (default 1)
    expected_states: to use for EM (in cases where gold-annotated states are unavailable)
    """
    
    def tokenize_truncate(self, inputs, mask, max_len,):
        """
        tensor (bsz, seqlen, *)
        mask (bsz)
        max_len (int)
        """
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=False)
        return {k: apply_mask_and_truncate(tokenized_inputs[k], mask, max_len, self.device) for k in tokenized_inputs}
    
    def collate_fn(self, batch):
        new_batch = {k: [] for k in batch[0]}
        for i, item in enumerate(batch):
            for k in item:
                new_batch[k].append(item[k])
        batch = new_batch
        
        # get context
        if self.control_input:
            control_context, control_mentions = [], []
            for entity in batch['entities']:
                control_context.append(str(entity)[1:-1])
                mention = [self.dataset.find_mention_in_ctxt(str(e), control_context[-1]) for e in entity]
                control_mentions.append(mention)
            context, batch['mentions'] = control_context, control_mentions
        else:
            context = batch['contexts']
        context_tokens = self.tokenizer(context, return_tensors='pt', padding=True, truncation=False, return_offsets_mapping=True)
        # get contexts within max length of model
        items_to_keep = context_tokens['attention_mask'].sum(1) <= self.tokenizer.model_max_length
        if not items_to_keep.any():
            return None, None, None, None, batch['game_ids'], None
        context_tokens = {k: apply_mask_and_truncate(context_tokens[k], items_to_keep, self.tokenizer.model_max_length, self.device) for k in context_tokens}
        # get lang tgts
        tgt_tokens = self.tokenize_truncate(batch['tgts'], items_to_keep, self.tokenizer.model_max_length)
        # get state tgts
        state_tokens = {}
        for state_type in ['init_states', 'tgt_states']:
            state_tokens[state_type] = {}
            for tf in ['true', 'false']:
                tokenized_state = self.tokenize_truncate([state[tf] for state in batch[state_type]], items_to_keep, self.tokenizer.model_max_length)
                state_token_key = f'{k}_{tf}' if tf == 'false' else k
                for k in tokenized_state: state_tokens[state_type][state_token_key] = tokenized_state[k]
        # mention sets/entity sets/gameids
        entity_sets = {
            'mentions': [ent for idx, ent in enumerate(batch['mentions']) if items_to_keep[idx]],
            'entities': [ent for idx, ent in enumerate(batch['entities']) if items_to_keep[idx]],
        }
        game_ids = [gid for idx, gid in enumerate(batch['game_ids']) if items_to_keep[idx]]

        # labels
        # (bs, # facts, seqlen[, embeddim])
        state_tokens['tgt_states']['all_states_input_ids'], state_tokens['tgt_states']['all_states_attn_mask'] = pad_stack(
            batch['all_states_tokenized'], pad_idx=self.tokenizer.pad_token_id, device=self.device,
        )
        state_tokens['tgt_states']['all_states_encoding'], encoding_mask = pad_stack(batch['all_states_encoded'], pad_idx=0, device=self.device)
        assert (encoding_mask == state_tokens['tgt_states']['all_states_attn_mask']).all()
        max_nfacts = state_tokens['tgt_states']['all_states_input_ids'].size(1)
        # (bs, # facts)
        labels = [lentry + [-1 for _ in range(max_nfacts-len(lentry))] for lentry in batch['labels']]
        labels = torch.tensor(labels).to(self.device)
        assert ((labels != -1) == state_tokens['tgt_states']['all_states_attn_mask'][:,:,0]).all()
        state_tokens['tgt_states']['labels'] = labels

        return context_tokens, tgt_tokens, state_tokens['init_states'], state_tokens['tgt_states'], game_ids, entity_sets


class TWFullDataLoader(DataLoader):
    def __init__(
        self, dataset: TWDataset, gamefile, tokenizer, batch_size, state_keys_to_get=[], max_gt_grounded_states=float("inf"), states=None,
        append_facts_to_input=False, include_feedback=False, nnegs=0, npos=1, device='cuda',
    ):
        """
        states: new set of states (must have 1 per sample for entire dataset) to override loaded states
        """
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        self.gamefile = gamefile
        self.state_keys_to_get = state_keys_to_get
        if len(state_keys_to_get) == 0:
            self.state_keys = []  # which field
            self.tgt_state_keys = []
            self.tgt_state_key = 'final_state'
        else:
            self.state_keys = [key[1].replace('_single', '').replace('_pair', '') for key in state_keys_to_get]
            self.tgt_state_keys = [key[0]+'_state' for key in state_keys_to_get]
        self.states = states
        self.nnegs = nnegs
        self.npos = npos
        self.append_facts_to_input = append_facts_to_input
        self.device = device
    
    def update_state(self, new_states):
        self.states = new_states

    def collate_fn(self, batch):
        game_ids = [item['filenames'].split('_')[0] for item in batch]
        contexts = [item['contexts'] for item in batch]
        context_tokens = self.tokenizer(contexts, return_tensors='pt', padding=True, truncation=False)
        items_to_keep = context_tokens['attention_mask'].sum(1) <= self.tokenizer.model_max_length
        if not items_to_keep.any():
            return None, None, None, None, game_ids, None

        # Delete problematic example(s) + truncate rest
        context_tokens = {key: apply_mask_and_truncate(context_tokens[key], items_to_keep, self.tokenizer.model_max_length, self.device) for key in context_tokens}
        tgts = [item['tgts'] for item in batch]
        tgt_tokens = self.tokenizer(tgts, return_tensors='pt', padding=True, truncation=True)
        # delete problem examples
        tgt_tokens = {key: apply_mask_and_truncate(tgt_tokens[key], items_to_keep, self.tokenizer.model_max_length, self.device) if type(tgt_tokens[key]) == torch.Tensor else tgt_tokens[key] for key in tgt_tokens}
        
        init_states = {}
        final_state = {}
        for sk, state_key in enumerate(self.state_keys):
            if 'belief_facts' in state_key:
                init_states[state_key] = {tf: [] for tf in batch[0]['init_state'][state_key]}
                final_state[state_key] = {tf: [] for tf in batch[0][self.tgt_state_keys[sk]][state_key]}
            elif 'full_facts' in state_key:
                init_states[state_key] = {'true': []}
                final_state[state_key] = {'true': []}
            ctxt = []
            for j in range(len(batch)):
                init_state_key = 'init_state'
                init_state, tgt_state = batch[j][init_state_key][state_key], batch[j][self.tgt_state_keys[sk]][state_key]
                if type(init_state) != dict: init_state, tgt_state = {'true': init_state}, {'true': tgt_state}
                env = textworld.start(os.path.join(self.gamefile, f'{game_ids[j-i]}.ulx'))
                game_state = env.reset()
                inform7_game = env._inform7
                for tf in init_state:
                    init_facts_gold = ' [SEP] '.join(parse_facts_to_nl(init_state[tf], inform7_game))
                    if j >= max_gt_grounded_states: init_facts = ''
                    else: init_facts = init_facts_gold
                    init_states[state_key][tf].append(init_facts)
                for tf in tgt_state:
                    tgt_facts_gold = ' [SEP] '.join(parse_facts_to_nl(tgt_state[tf], inform7_game))
                    if j >= max_gt_grounded_states:
                        tgt_facts = ''
                    else:
                        tgt_facts = tgt_facts_gold
                    final_state[state_key][tf].append(tgt_facts)
                    
        init_state_tokens = {}
        tgt_state_tokens = {}
        for state_key in init_states:
            init_state_tokens[state_key] = {}
            tgt_state_tokens[state_key] = {}
            for tf in init_states[state_key]:
                tokenized_init_tf = self.tokenizer(init_states[state_key][tf], return_tensors='pt', padding=True, truncation=True).to(self.device)
                tokenized_tgt_tf = self.tokenizer(final_state[state_key][tf], return_tensors='pt', padding=True, truncation=True).to(self.device)
                for k2 in tokenized_init_tf:
                    init_state_tokens[state_key][f'{k2}{"_"+tf if tf != "true" else ""}'] = tokenized_init_tf[k2]
                    tgt_state_tokens[state_key][f'{k2}{"_"+tf if tf != "true" else ""}'] = tokenized_tgt_tf[k2]
        game_ids = [gid for gidx, gid in enumerate(game_ids) if items_to_keep[gidx]]
        return context_tokens, tgt_tokens, init_state_tokens, tgt_state_tokens, game_ids, 'all'
