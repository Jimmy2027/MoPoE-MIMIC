import numpy as np
import torch
from typing import List, Iterable, Union

digit_text_german = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun']
digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq: int, alphabet: str, seq: str) -> torch.tensor:
    """
    One hot encodes the sequence.
    Set $ for the end of text. Pads with & to len_seq. Replaces chars that are not found in the alphabet with @.
    len_seq is the maximum sequence length

    """
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq]
    elif len(seq) < len_seq:
        seq += '$'
        seq = seq.ljust(len_seq, '&')

    for index_char, char in enumerate(seq):
        # char2Index return -1 if the char is not found in the alphabet.
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
        else:
            X[index_char, alphabet.find('@')] = 1.0

    return X


def seq2text(exp, seq: Iterable[int]) -> List[str]:
    """
    seg: list of indices
    """
    decoded = []
    for j in range(len(seq)):
        if exp.flags.text_encoding == 'word':
            decoded.append(exp.dataset_train.report_findings_dataset.i2w[str(int(seq[j]))])
        else:
            decoded.append(exp.alphabet[seq[j]])
    return decoded


def tensor_to_text(exp, gen_t: torch.Tensor, one_hot=True) -> Union[List[List[str]], List[str]]:
    """
    Converts a one hot encoded tensor or an array of indices to sentences
    gen_t: tensor of shape (bs, length_sent, num_features) if one_hot else (bs, length_sent)
    one_hot: if one_hot is True, gen_t needs to be a one-hot-encoded matrix. The maximum along every axis is taken
    to create a list of indices.
    """
    gen_t = gen_t.cpu().data.numpy()
    if one_hot:
        gen_t = np.argmax(gen_t, axis=-1)
        gen_t: np.ndarray
    if len(gen_t.shape) == 1:
        return seq2text(exp, gen_t)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(exp, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples
