import sys
import os
import json

import torch

from run_epochs import run_epochs

from utils.filehandling import create_dir_structure
from mimic.flags import parser
from mimic.experiment import MimicExperiment

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
    elif FLAGS.method == 'joint_elbo':
        FLAGS.joint_elbo=True;
    else:
        print('method implemented...exit!')
        sys.exit();
    print(FLAGS.modality_poe)
    print(FLAGS.modality_moe)
    print(FLAGS.modality_jsd)
    print(FLAGS.joint_elbo)

    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content];

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mimic = MimicExperiment(FLAGS, alphabet);
    create_dir_structure(mimic)
    mimic.set_optimizer();

    run_epochs(mimic);
