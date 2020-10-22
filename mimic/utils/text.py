import random

import numpy as np
import torch

# digit_text = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_german = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq: int, alphabet: str, seq: str) -> torch.tensor:
    """
    One hot encodes the sequence.
    len_seq is the maximum sequence length
    """
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq];
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def one_hot_encode_word(args, seq: torch.tensor) -> torch.tensor:
    """
    One hot encodes word encodings.
    Example: for tensor of shape (bs, 1024)
    returns tensor of shape (bs, 1024, vocab_size)
    """
    if len(seq.shape) > 1:
        # need to iterate over batches
        X = torch.zeros(seq.shape[0], args.len_sequence, args.vocab_size)
        for sentence in range(seq.shape[0]):
            for index_char, char in enumerate(seq[sentence]):
                X[sentence, index_char, int(char.item())] = 1.0
    else:
        X = torch.zeros(args.len_sequence, args.vocab_size)
        for index_char, char in enumerate(seq):
            X[index_char, int(char.item())] = 1.0
    return X.to(args.device)


def seq2text(exp, seq):
    decoded = []
    for j in range(len(seq)):
        if exp.flags.text_encoding == 'word':
            decoded.append(exp.dataset_train.report_findings_dataset.i2w[str(seq[j])])
        else:
            decoded.append(exp.alphabet[seq[j]])
    return decoded


def tensor_to_text(exp, gen_t: torch.Tensor):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(exp, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples;
