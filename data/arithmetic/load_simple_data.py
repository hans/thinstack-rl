
#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )

import numpy as np

from util.data import PADDING_TOKEN
import util.data
import pprint

SENTENCE_PAIR_DATA = False

NUMBERS = range(-10, 11)
OUTPUTS = range(-100, 101)
NUM_CLASSES = len(OUTPUTS)

FIXED_VOCABULARY = {str(x): x + min(NUMBERS) for x in NUMBERS}
FIXED_VOCABULARY.update({
    PADDING_TOKEN: len(FIXED_VOCABULARY) + 0,
    "+": len(FIXED_VOCABULARY) + 1,
    "-": len(FIXED_VOCABULARY) + 2
})
assert len(set(FIXED_VOCABULARY.values())) == len(FIXED_VOCABULARY.values())

LABEL_MAP = {str(x): x - min(OUTPUTS) for x in OUTPUTS}


def convert_binary_bracketed_data(filename):
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            tab_split = line.split('\t')
            example["label"] = tab_split[0]
            example["sentence"] = tab_split[1]
            example["tokens"] = []
            example["transitions"] = []

            for word in example["sentence"].split(' '):
                if word != "(":
                    if word != ")":
                        example["tokens"].append(word)
                    example["transitions"].append(1 if word == ")" else 0)

            examples.append(example)
    return examples


def load_data(path):
    dataset = convert_binary_bracketed_data(path)
    return dataset, FIXED_VOCABULARY

if __name__ == "__main__":
    # Demo:
    examples, _ = load_data('data/arithmetic/test/simple_test.tsv')
    examples = util.data.PadAndBucket(examples, [9, 17])

    pprint.pprint(examples)
