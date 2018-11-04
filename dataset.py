import unicodedata
import re
import random
import torch
from torch.utils.data import Dataset

PAD = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)



class Seq2SeqDataset(Dataset):

    def __init__(self, pairs, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.input_seq = torch.zeros((len(pairs)), MAX_LENGTH, dtype=torch.int64)
        self.label_seq = torch.zeros((len(pairs)), MAX_LENGTH, dtype=torch.int64)
        self.lengths_input = []
        self.lengths_output = []

        for i, pair in enumerate(pairs):
            lang1_sample, lang2_sample = tensors_from_pair(pair, input_lang, output_lang)

            self.lengths_input.append(len(lang1_sample))
            self.lengths_output.append(len(lang2_sample))

            self.input_seq[i, :len(lang1_sample)] = lang1_sample.squeeze_()
            self.label_seq[i, :len(lang2_sample)] = lang2_sample.squeeze_()

        print('hi')

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        return self.input_seq[idx], self.label_seq[idx]


class Lang:
    """ Helper class to create and store dictionary of our vocabulary."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count PAD, SOS and EOS.

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # Lowercase, trim, and remove non-letter characters.
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines.
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize.
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances.
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(pair, reverse):
    return len(pair[0].split(' ')) < MAX_LENGTH and \
           len(pair[1].split(' ')) < MAX_LENGTH and \
           pair[1 if reverse else 0].startswith(eng_prefixes)


def filter_pairs(pairs, reverse):
    return [pair for pair in pairs if filter_pair(pair, reverse)]


def prepare_data(lang1, lang2, reverse):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs, reverse)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensors_from_pair(pair, input_lang, output_lang):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


def main():
    # Test data loading.
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', False)
    print(random.choice(pairs))


if __name__ == "__main__":
    main()
