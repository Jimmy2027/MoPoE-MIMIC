import sys
import random

import numpy as np
import torch
import torch.nn.functional as F

# digit_text = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_german = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

face_text = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald' 'Bangs' 'Big_Lips',
             'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
             'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
             'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
             'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
             'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie' 'Young']

def char2Index(alphabet, character):
    return alphabet.find(character)

def one_hot_encode(len_seq, alphabet, seq):
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq];
    for index_char, char in enumerate(seq[::-1]):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def create_text_from_target_mnist(args, target, alphabet):
    X = torch.zeros(args.batch_size, args.len_sequence, len(alphabet))
    for k in range(0, len(target)):
        num = target[k];
        text = digit_text_english[num];
        sequence = args.len_sequence*[' '];
        start_index = random.randint(0, args.len_sequence-1);
        sequence[start_index:start_index+len(text)] = text;
        sequence_one_hot = one_hot_encode(args, alphabet, sequence);
        X[k,:,:] = sequence_one_hot;
    return X;


def create_text_from_label_mnist(len_seq, label, alphabet):
    text = digit_text_english[label];
    sequence = len_seq * [' '];
    start_index = random.randint(0, len_seq - 1 - len(text));
    sequence[start_index:start_index + len(text)] = text;
    sequence_one_hot = one_hot_encode(len_seq, alphabet, sequence);
    return sequence_one_hot


# def create_text_from_label_mnist(args, label, alphabet):
#     text = digit_text[label];
#     sequence = args.len_sequence * [' '];
#     start_index = random.randint(0, args.len_sequence - 1 - len(text));
#     sequence[start_index:start_index + len(text)] = text;
#     sequence_one_hot = one_hot_encode(args, alphabet, sequence);
#     return sequence_one_hot

def calc_reconstruct_error_text(args, log_prob, target):
    if args.output_activation == 'softmax':
        reconstruct_error_text = F.binary_cross_entropy_with_logits(log_prob, target, size_average=False) / float(args.batch_size);
    elif args.output_activation == 'sigmoid':
        if args.loss_function_text == 'binary_cross_entropy':
            reconstruct_error_text = F.binary_cross_entropy(log_prob, target, size_average=False) / float(args.batch_size);
        elif args.loss_function_text == 'mse':
            reconstruct_error_text = F.mse_loss(log_prob, target, reduction='sum') / float(args.batch_size);
        else:
            print('loss function not implemented...')
            sys.exit()
    return reconstruct_error_text;


def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded

def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(tuple(decoded))
    return decoded_samples;


def write_samples_text_to_file(samples, filename):
    file_samples = open(filename, 'w');
    for k in range(0, len(samples)):
        file_samples.write(''.join(samples[k])[::-1] + '\n');
    file_samples.close();
