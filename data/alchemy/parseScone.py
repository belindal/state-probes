# from https://worksheets.codalab.org/worksheets/0xad3fc9f52f514e849b282a105b1e3f02


class SceneDataset:
    def __init__():
        self.data = []

    def addLine(self, ln):
        pass 


def parse_state(lst, kind='scene'):
    #given a sequence of strings, whcih string to use?
    #we will ignore the numbers
    if kind=='scene':
        return [pos[-2:] for pos in lst]
    elif kind =='alchemy':
        return lst
    elif kind == 'tangram':
        assert False
    else: assert False

def parse_synth_state(lst, kind='scene'):
    #given a sequence of strings, whcih string to use?
    #we will ignore the numbers
    if kind=='scene':
        assert False
    elif kind =='alchemy':
        return [str(i+1) + ':' + obj for i, obj in enumerate(lst)]
    elif kind == 'tangram':
        assert False
    else: assert False


class Datum:
    def __init__(self, ln, kind='scene'):
        self.kind = kind
        line = ln.split('\t')[1:] #remove first thing
        self.initial_state = parse_state(line[0].split(' '), kind=kind)
        self.statements = [ item.split(' ') for i, item in enumerate(line[1:]) if i%2==0]
        self.states = [ parse_state(item.split(' '), kind=kind) for i, item in enumerate(line[1:]) if i%2==1]

    def proximal_lang_pairs():
        for i in range(len(self.statements) -1):
            yield self.statements[i], self.statements[i+1]

    def proximal_state_pairs():
        for statement, state in zip(self.statements, self.states):
            return statement, state

    def all_pairs(self, join='\n'):
        for i in range(len(self.statements) -1):
            prior = []
            for j in range(i+1):
                if j>0: prior.append(join)
                prior.extend(self.statements[j])
            yield prior, self.statements[i+1], self.states[i]

class SyntheticDatum(Datum):
    def __init__(self, state_ln, lang_ln, kind='alchemy'):
        self.kind = kind
        #line = ln.split('\t')[1:] #remove first thing
        states = state_ln[:-1].split(',')[1:]
        sentences = lang_ln.split(',')[1].split(". ")
        sentences = [s for s in sentences if s != '\n']

        self.initial_state = parse_synth_state(states[0].split(' '), kind=kind) #can add numbers here if i want
        self.statements = [ item.split(' ') for item in sentences]
        self.states = [ parse_synth_state(item.split(' '), kind=kind) for item in states[1:]]

def get_lang_vocab(dataset):
    vocab = set()
    for datum in dataset:
        for statement in datum.statements:
            for word in statement:
                vocab.add(word)
    return vocab

def get_state_vocab(dataset):
    vocab = set()
    for datum in dataset:
        for state in datum.states:
            for word in state:
                vocab.add(word)
    return vocab

def loadData(kind='scene', split="train", synthetic=False):
    if synthetic: return loadSyntheticData(kind=kind, split=split)
    path = f"rlong/{kind}-{split}.tsv"  

    dataset = []

    with open(path, 'r') as f:
        for ln in f:
            dataset.append(Datum(ln, kind=kind))

    return dataset, get_lang_vocab(dataset), get_state_vocab(dataset)

def loadSyntheticData(kind='scene', split='train'):
    state_path = f"synth_{kind}/{split}/{split}_answers.csv"
    lang_path = f"synth_{kind}/{split}/batch.csv"
    dataset = []
    with open(state_path, 'r') as f:
        with open(lang_path, 'r') as g:
            for i, (state_ln, lang_ln) in enumerate(zip(f,g)):
                if i>0: dataset.append(SyntheticDatum( state_ln, lang_ln  ))

    return dataset, get_lang_vocab(dataset), get_state_vocab(dataset)




def getBatches(dataset, batchsize):
    pairs = [x for d in dataset for x in d.all_pairs()] 
    pairs = sorted(pairs, key=lambda x: len(x[0]))
    batch = []
    for i, pair in enumerate(pairs):
        if i!=0 and i%batchsize==0:
            yield batch
            batch = []
        batch.append(pair)
    yield batch

def getBatchesWithInit(dataset, batchsize):
    pairs = [x + (d.initial_state,) for d in dataset for x in d.all_pairs()] 
    pairs = sorted(pairs, key=lambda x: len(x[0]))
    batch = []
    for i, pair in enumerate(pairs):
        if i!=0 and i%batchsize==0:
            yield batch
            batch = []
        batch.append(pair)
    yield batch

def getBatches(dataset, batchsize):
    pairs = [x for d in dataset for x in d.all_pairs()] 
    pairs = sorted(pairs, key=lambda x: len(x[0]))
    batch = []
    for i, pair in enumerate(pairs):
        if i!=0 and i%batchsize==0:
            yield batch
            batch = []
        batch.append(pair)
    yield batch

def getRBBatches(dataset, batchsize):
    for batch in getBatches(dataset, batchsize):
        inputs, lang_targets, state_targets = zip(*batch)
        inps = [ [inp_seq ] for inp_seq in inputs]
        yield inps, lang_targets, state_targets

def getSeq2SeqBatches(dataset, batchsize):
    for batch in getBatches(dataset, batchsize):
        inputs, lang_targets, state_targets = zip(*batch)
        pass


def prepareData(pairs, verbose=True):
    from data_loader_batch import Lang
    # Input
    #  fn_in : input file name (or list of file names to concat.)
    #
    #  Read text file and split into lines, split lines into pairs
    #  Make word lists from sentences in pairs
    if verbose: print("Processing input data")
    if verbose: print(" Reading lines...")
    word_lang = Lang('word')
    state_lang = Lang('state')

    if verbose:
        print(" Read %s sentence pairs" % len(pairs))
        print(" Counting words...")
    for pair in pairs:
        for word in pair[0]: word_lang.addWord(word)
        for word in pair[1]: word_lang.addWord(word) 
        for word in pair[2]: state_lang.addWord(word) 
    if verbose:
        print(" Counted words:")
        print(' ',word_lang.name, word_lang.n_words)
        print(' ',state_lang.name, state_lang.n_words)
        print('')
    return word_lang, state_lang, pairs



if __name__ =='__main__':
    #dataset, lang_v, state_v = loadData(split="train")
    kind="alchemy"

    dataset, lang_v, state_v = loadSyntheticData(kind=kind, split="train")

    _, lang_v_dev, state_v_dev = loadSyntheticData(kind=kind, split="dev")
    _, lang_v_test, state_v_test = loadSyntheticData(kind=kind, split="test")

    print(lang_v == lang_v_test)
    print(lang_v == lang_v_dev)

    pairs = [x for d in dataset for x in d.all_pairs()] 
    pairs = sorted(pairs, key=lambda x: len(x[0]))
    words = [word for d in dataset for statement in d.statements for word in statement]