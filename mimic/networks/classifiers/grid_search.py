from sklearn.model_selection import ParameterGrid

from mimic.networks.classifiers.main_train_clf_mimic import run_training_procedure_clf
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config

parser.add_argument('--which_grid', type=str, default='text', choices=['resnet', 'densenet', 'text'],
                    help="modality on which to train the image classifier, chose between PA and Lateral")

params_seach_space_densenet = {
    'n_crops': [1, 5, 10],
    'img_clf_type': ['densenet'],
    'clf_loss': ['bce_with_logits', 'binary_crossentropy'],
    'img_size': [256],
    'modality': ['PA', 'Lateral']
}

params_seach_space_resnet = {
    'img_clf_type': ['resnet'],
    'clf_loss': ['bce_with_logits', 'binary_crossentropy'],
    'img_size': [256, 128],
    'modality': ['PA', 'Lateral']
}

params_seach_space_text = {
    'text_encoding': ['char', 'word'],
    'clf_loss': ['bce_with_logits', 'binary_crossentropy'],
    'modality': ['text']
}

blacklist = [
    {'img_clf_type': ['resnet'],
     'clf_loss': ['bce_with_logits'],
     'img_size': [256],
     'modality': ['PA']}
]

grids = {
    'resnet': params_seach_space_resnet,
    'densenet': params_seach_space_densenet,
    'text': params_seach_space_text
}

FLAGS = parser.parse_args()
config_path = get_config_path()

for params in ParameterGrid(grids[FLAGS.which_grid]):
    if params not in blacklist:
        print(params)
        FLAGS = update_flags_with_config(config_path, additional_args=params)

        FLAGS.batch_size = 500
        FLAGS.max_early_stopping_index = 10

        assert FLAGS.n_crops in [1, 5, 10]
        assert FLAGS.img_clf_type in ['densenet', 'resnet']
        assert FLAGS.clf_loss in ['bce_with_logits', 'crossentropy', 'binary_crossentropy']

        FLAGS.dir_clf += '_gridsearch'
        FLAGS.distributed = False
        run_training_procedure_clf(FLAGS)
