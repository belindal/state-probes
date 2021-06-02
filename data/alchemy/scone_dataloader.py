from data.alchemy.utils import (
    int_to_word, decide_translate, translate_states_to_nl, translate_states_to_nl,
)
from data.alchemy.parse_alchemy import (
    parse_utt_with_world, parse_world,
)
from data.alchemy.parseScone import getBatchesWithInit
from data.alchemy.alchemy_artificial_generator import execute
from transformers import BartTokenizerFast, T5TokenizerFast


def convert_to_transformer_batches(
    dataset, tokenizer, batchsize, random=None, include_init_state=False,
    no_context=False, append_last_state_to_context=False, domain="alchemy",
    state_targets_type=None, device='cuda', control_input=False,
):
    """
    state_targets_type (str): what to return for `state_tgt_enc` and `state_targets`
        {state|init_state|interm_states|single_beaker_{...}|utt_beaker_{...}{.offset{1|2|...|6}}}.{NL|raw}|text
    """
    if state_targets_type is None:
        state_targets_type = "state.NL"
    state_targets_type_split = state_targets_type.split('.')
    batches = list(getBatchesWithInit(dataset, batchsize))
    if random: random.shuffle(batches)
    for batch in batches:
        inputs, lang_targets, state_targets, init_states = zip(*batch)

        init_states = [' '.join(init_state) for init_state in init_states]

        # make state targets
        if state_targets_type_split[0] == 'state':
            state_targets = [decide_translate(' '.join(tgt), state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast)) for tgt in state_targets]
            state_targets = {'full_state': state_targets}
        elif state_targets_type_split[0] == 'init_state':
            state_targets = [decide_translate(init_state, state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast)) for init_state in init_states]
            state_targets = {'full_state': state_targets}
        elif state_targets_type_split[0].startswith('single_beaker'):
            assert domain == 'alchemy'
            if state_targets_type_split[0].endswith("init"):
                states = init_states
            elif state_targets_type_split[0].endswith("final"):
                states = [' '.join(tgt) for tgt in state_targets]
            else:
                raise NotImplementedError()
            # {bn -> [`bn`-th beaker's state in example i (x # examples for beaker bn)]}
            state_targets = {int(beaker.split(':')[0]) - 1: [] for beaker in states[0].split(' ')}
            for state in states:
                state_descr = decide_translate(state, state_targets_type, domain, isinstance(tokenizer, BartTokenizerFast))
                if isinstance(tokenizer, BartTokenizerFast):
                    beaker_states = state_descr.split(',')
                else:
                    beaker_states = state_descr.split(', ')
                for bn, beaker_state in enumerate(beaker_states):
                    state_targets[bn].append(beaker_state)
        else: assert state_targets_type_split[0] == 'text'

        # make inputs
        inps = []
        for i, inp in enumerate(inputs):
            if no_context:
                string = ''
            else:
                if isinstance(tokenizer, T5TokenizerFast):
                    string = ' '.join(inp).replace(' \n ', '. ')
                    if '  ' in string: string = string.replace('  ', ' first ')
                else:
                    string = ' '.join(inp).replace(' \n ', '.\n')
                if include_init_state == "raw":
                    string = init_states[i] + '. ' + string
                elif include_init_state == "NL":
                    # translate to natural language
                    string = translate_states_to_nl(init_states[i], domain, isinstance(tokenizer, BartTokenizerFast)) + '. ' + string
                else:
                    assert not include_init_state
                if append_last_state_to_context:
                    string += state_targets[i]
            inps.append(string)
        if state_targets_type_split[0] == 'text':
            state_targets = {'original_text': inps}

        # make lang targets
        lang_targets_new = []
        for tgt in lang_targets:
            tgt = ' '.join(tgt) + '.'
            if isinstance(tokenizer, T5TokenizerFast) and '  ' in tgt:
                tgt = tgt.replace('  ', ' first ')
            lang_targets_new.append(tgt)
        lang_targets = lang_targets_new

        if control_input:
            # the first beaker has 1 red, the second beaker has 1 red 
            inps = [", ".join([f"the {int_to_word[num]} beaker is empty" for num in range(7)]) + "." for _ in inps]
        inp_enc = tokenizer(inps, return_tensors='pt', padding=True, truncation=True, return_offsets_mapping=True).to(device)
        lang_tgt_enc = tokenizer(lang_targets, return_tensors='pt', padding=True, truncation=True).to(device)
        inp_enc['original_text'] = inps
        lang_tgt_enc['original_text'] = lang_targets
        for state_key in state_targets:
            state_tgt_enc = tokenizer(state_targets[state_key], return_tensors='pt', padding=True, truncation=True).to(device)
            inp_enc['state_key'] = state_key
            state_tgt_enc['key'] = state_key
            yield inp_enc, lang_tgt_enc, state_tgt_enc, state_targets, init_states
