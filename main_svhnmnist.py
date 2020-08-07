import sys
import os
import json

import torch

from mnistsvhntext.training import run_epochs

from utils.filehandling import create_dir_structure
from mnistsvhntext.flags import parser
from mnistsvhntext.experiment import MNISTSVHNText

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    if FLAGS.method == 'poe':
        FLAGS.modality_poe=True;
        FLAGS.poe_unimodal_elbos=True;
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe=True;
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd=True;
    else:
        print('method implemented...exit!')
        sys.exit();

    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content];
    create_dir_structure(FLAGS)

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mst = MNISTSVHNText(FLAGS, alphabet);
    mst.set_optimizer();

    run_epochs(mst);
