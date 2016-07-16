"""Dataset handling and related yuck."""

import random
import itertools

import numpy as np


# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "*PADDING*"

# Temporary hack: Map UNK to "_" when loading pretrained embedding matrices:
# it's a common token that is pretrained, but shouldn't look like any content words.
UNK_TOKEN = "_"

CORE_VOCABULARY = {PADDING_TOKEN: 0,
                   UNK_TOKEN: 1}

# Allowed number of transition types : currently PUSH : 0 and MERGE : 1
NUM_TRANSITION_TYPES = 2


# Keys into transition sequences / tokens / sequence lengths for different
# dataset types. Internal use only.
_SENTENCE_PAIR_KEYS = [("premise_transitions", "premise_tokens", "premise_len"),
                       ("hypothesis_transitions", "hypothesis_tokens", "hypothesis_len")]
_SENTENCE_KEYS = [("transitions", "tokens", "len")]


def TrimDataset(dataset, seq_length, eval_mode=False, sentence_pair_data=False):
    """Avoid using excessively long training examples."""
    if eval_mode:
        return dataset
    else:
        if sentence_pair_data:
            new_dataset = [example for example in dataset if
                len(example["premise_transitions"]) <= seq_length and
                len(example["hypothesis_transitions"]) <= seq_length]
        else:
            new_dataset = [example for example in dataset if len(
                example["transitions"]) <= seq_length]
        return new_dataset


def TokensToIDs(vocabulary, dataset, sentence_pair_data=False):
    """Replace strings in original boolean dataset with token IDs."""
    if sentence_pair_data:
        keys = ["premise_tokens", "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for key in keys:
        if UNK_TOKEN in vocabulary:
            unk_id = vocabulary[UNK_TOKEN]
            for example in dataset:
                example[key] = [vocabulary.get(token, unk_id)
                                for token in example[key]]
        else:
            for example in dataset:
                example[key] = [vocabulary[token]
                                for token in example[key]]
    return dataset


def PadExample(example, actual_transitions, desired_transitions, keys):
    assert actual_transitions <= desired_transitions
    num_tokens = (desired_transitions + 1) / 2

    for transitions_key, tokens_key, len_key in keys:
        example[len_key] = actual_transitions

        # Pad everything at right.
        example[transitions_key] += [0] * (desired_transitions - len(example[transitions_key]))
        example[tokens_key] += [0] * (num_tokens - len(example[tokens_key]))

    return example


def PadDataset(dataset, desired_length, sentence_pair_data=False):
    keys = _SENTENCE_PAIR_KEYS if sentence_pair_data else _SENTENCE_KEYS

    for example in dataset:
        max_transitions = max(len(example[transitions_key]) for transitions_key, _, _ in keys)
        assert max_transitions <= desired_length
        PadExample(example, max_transitions, desired_length, keys)

    return dataset


def PadAndBucket(dataset, lengths, batch_size, sentence_pair_data=False):
    """Pad sequences and categorize them into different length bins."""
    keys = _SENTENCE_PAIR_KEYS if sentence_pair_data else _SENTENCE_KEYS

    for length in lengths:
        if length % 2 == 0:
            raise ValueError("only odd sequence lengths are valid for SR transitions. %s" % lengths)

    lengths = sorted(lengths)
    buckets = {length: [] for length in lengths}
    def get_nearest_bucket(length):
        for bucket_length in lengths:
            if length <= bucket_length:
                return bucket_length
        raise ValueError("length %i is larger than largest bucket length %i"
                         % (length, lengths[-1]))

    for example in dataset:
        # For a sentence-pair dataset, the longer of the two sentences
        # determines in which bucket the pair goes :(
        max_transitions = max(len(example[transitions_key]) for transitions_key, _, _ in keys)

        nearest_bucket = get_nearest_bucket(max_transitions)
        example = PadExample(example, max_transitions, nearest_bucket, keys)

        buckets[nearest_bucket].append(example)

    for bucket in buckets:
        assert len(buckets[bucket]) >= batch_size, \
                "Bucket smaller than batch size: %s (only %i examples)" % (bucket, len(buckets[bucket]))

    return buckets


def CropAndPadForRNN(dataset, length, logger=None, sentence_pair_data=False):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    if sentence_pair_data:
        keys = ["premise_tokens",
                "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for example in dataset:
        for tokens_key in keys:
            num_tokens = len(example[tokens_key])
            tokens_left_padding = length - num_tokens
            CropAndPadExample(
                example, tokens_left_padding, length, tokens_key, logger=logger)
    return dataset


def MakeTrainingIterator(sources, batch_size):
    # Make an iterator that exposes a dataset as random minibatches.

    def data_iter():
        dataset_size = len(sources[0])
        start = -1 * batch_size
        order = range(dataset_size)
        random.shuffle(order)

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            yield tuple(source[batch_indices] for source in sources)
    return data_iter()


def MakeBucketedTrainingIterator(bucketed_sources, batch_size, selector=None):
    if selector is None:
        buckets = bucketed_sources.keys()
        total_size = sum([len(bucketed_sources[bucket][1]) for bucket in buckets])
        bucket_sampling_p = [float(len(bucketed_sources[bucket][1]))/total_size
                             for bucket in buckets]
        def selector():
            bucket_idx = np.random.choice(len(buckets), p=bucket_sampling_p)
            return buckets[bucket_idx]

    iterators = {length: MakeTrainingIterator(bucket, batch_size)
                 for length, bucket in bucketed_sources.iteritems()}
    while True:
        choice = selector()
        yield choice, next(iterators[choice])


def MakeEvalIterator(sources, batch_size):
    # Make a list of minibatches from a dataset to use as an iterator.
    # TODO(SB): Pad out the last few examples in the eval set if they don't
    # form a batch.

    print "WARNING: May be discarding eval examples."

    dataset_size = len(sources[0])
    data_iter = []
    start = -batch_size
    while True:
        start += batch_size

        if start >= dataset_size:
            break

        candidate_batch = tuple(source[start:start + batch_size]
                               for source in sources)

        if len(candidate_batch[0]) == batch_size:
            data_iter.append(candidate_batch)
        else:
            print "Skipping " + str(len(candidate_batch[0])) + " examples."
    return data_iter


def BucketToArrays(dataset, data_manager, for_rnn=False):
    if data_manager.SENTENCE_PAIR_DATA:
        X = np.array([[example["premise_tokens"] for example in dataset],
                      [example["hypothesis_tokens"] for example in dataset]],
                     dtype=np.int32)
        if for_rnn:
            # TODO(SB): Extend this clause to the non-pair case.
            transitions = np.zeros((len(dataset), 2, 0))
            num_transitions = np.zeros((len(dataset), 2))
        else:
            transitions = np.array([[example["premise_transitions"] for example in dataset],
                                    [example["hypothesis_transitions"] for example in dataset]],
                                   dtype=np.int32)
            num_transitions = np.array(
                [[example["premise_len"] for example in dataset],
                 [example["hypothesis_len"] for example in dataset]],
                dtype=np.int32)
    else:
        X = np.array([[example["tokens"] for example in dataset]],
                     dtype=np.int32)
        transitions = np.array([[example["transitions"] for example in dataset]],
                               dtype=np.int32)
        num_transitions = np.array([[example["len"] for example in dataset]],
                                   dtype=np.int32)

    # Transpose from (num_stacks, num_examples, num_timesteps)
    # to (num_examples, num_stacks, num_timesteps)
    X = np.transpose(X, (1, 0, 2))
    transitions = np.transpose(transitions, (1, 0, 2))
    # transpose from (num_stacks, num_examples)
    # to (num_examples, num_stacks)
    num_transitions = num_transitions.T

    y = np.array(
        [data_manager.LABEL_MAP[example["label"]] for example in dataset],
        dtype=np.int32)

    return X, transitions, num_transitions, y


def BuildVocabulary(raw_training_data, raw_eval_sets, embedding_path, sentence_pair_data=False):
    # Find the set of words that occur in the data.
    types_in_data = set()
    for dataset in [raw_training_data] + [eval_dataset[1] for eval_dataset in raw_eval_sets]:
        if sentence_pair_data:
            types_in_data.update(itertools.chain.from_iterable([example["premise_tokens"]
                                                                for example in dataset]))
            types_in_data.update(itertools.chain.from_iterable([example["hypothesis_tokens"]
                                                                for example in dataset]))
        else:
            types_in_data.update(itertools.chain.from_iterable([example["tokens"]
                                                                for example in dataset]))

    if embedding_path == None:
        vocabulary = CORE_VOCABULARY
    else:
        # Build a vocabulary of words in the data for which we have an
        # embedding.
        vocabulary = BuildVocabularyForASCIIEmbeddingFile(
            embedding_path, types_in_data, CORE_VOCABULARY)

    return vocabulary


def BuildVocabularyForASCIIEmbeddingFile(path, types_in_data, core_vocabulary):
    """Quickly iterates through a GloVe-formatted ASCII vector file to
    extract a working vocabulary of words that occur both in the data and
    in the vector file."""

    # TODO(SB): Report on *which* words are skipped. See if any are common.

    vocabulary = {}
    vocabulary.update(core_vocabulary)
    next_index = len(vocabulary)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ", 1)
            word = spl[0]
            if word in types_in_data:
                vocabulary[word] = next_index
                next_index += 1
    return vocabulary


def LoadEmbeddingsFromASCII(vocabulary, embedding_dim, path):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a GloVe - format ASCII vector file.

    For now, values not found in the file will be set to zero."""
    emb = np.zeros(
        (len(vocabulary), embedding_dim), dtype=np.float32)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ")
            word = spl[0]
            if word in vocabulary:
                emb[vocabulary[word], :] = [float(e) for e in spl[1:]]
    return emb


def TransitionsToParse(transitions, words):
    if transitions is not None:
        stack = ["(P *ZEROS*)"] * (len(transitions) + 1)
        buffer_ptr = 0
        for transition in transitions:
            if transition == 0:
                stack.append("(P " + words[buffer_ptr] +")")
                buffer_ptr += 1
            elif transition == 1:
                r = stack.pop()
                l = stack.pop()
                stack.append("(M " + l + " " + r + ")")
        return stack.pop()
    else:
        return " ".join(words)
