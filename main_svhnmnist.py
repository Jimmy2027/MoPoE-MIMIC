import sys
from mnistsvhntext.training_svhnmnist import training_svhnmnist

from utils.filehandling import create_dir_structure
from mnistsvhntext.flags_svhnmnist import parser

if __name__ == '__main__':
    FLAGS = parser.parse_args()

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
    print(FLAGS.factorized_representation)
    training_svhnmnist(FLAGS);
