from sklearn.model_selection import ParameterGrid

from mimic.networks.classifiers.main_train_clf_mimic import run_training_procedure_clf
from mimic.utils.filehandling import get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import update_flags_with_config
from mimic import log

parser.add_argument('--which_grid', type=str, default='imgs', choices=['resnet', 'imgs', 'text'],
                    help="modality on which to train the image classifier, chose between PA and Lateral")

params_seach_space_img = {
    'exp_str_prefix': [['weighted_sampler', 'img_clf_type', 'initial_learning_rate', 'binary_labels']],
    'n_crops': [1],
    'img_clf_type': ['resnet'],
    'clf_loss': ['dice'],
    'img_size': [256],
    'modality': ['PA', 'Lateral'],
    'fixed_extractor': [True],
    'binary_labels': [True, False],
    'normalization': [False],
    'weighted_sampler': [True],
    "undersample_dataset": [False],
    "initial_learning_rate": [0.0001]
}

params_seach_space_PA = {
    'img_clf_type': ['densenet'],
    'clf_loss': ['binary_crossentropy'],
    'img_size': [128],
    'modality': ['Lateral']
}

params_seach_space_text = {
    'exp_str_prefix': [['weighted_sampler', 'initial_learning_rate', 'binary_labels']],
    'text_encoding': ['word'],
    'clf_loss': ['binary_crossentropy'],
    'modality': ['text'],
    'reduce_lr_on_plateau': [True],
    'binary_labels': [True],
    'normalization': [False],
    'weighted_sampler': [True],
    "initial_learning_rate": [0.0001]
}

blacklist = [
    {'img_clf_type': ['resnet'],
     'clf_loss': ['bce_with_logits'],
     'img_size': [256],
     'modality': ['PA']}
]

grids = {
    'imgs': params_seach_space_img,
    'text': params_seach_space_text
}

FLAGS = parser.parse_args()
config_path = get_config_path()

for params in ParameterGrid(grids[FLAGS.which_grid]):
    if params not in blacklist:

        if 'exp_str_prefix' in params:
            params['exp_str_prefix'] = '_'.join(f'{k}-{params[k]}' for k in params['exp_str_prefix'])

        log.info(f'{params}')
        FLAGS = update_flags_with_config(config_path, additional_args=params)

        FLAGS.batch_size = 100
        FLAGS.max_early_stopping_index = 10

        assert FLAGS.n_crops in [1, 5, 10]
        assert FLAGS.img_clf_type in ['densenet', 'resnet']
        assert FLAGS.clf_loss in ['dice', 'crossentropy', 'binary_crossentropy'], f'{FLAGS.clf_loss} not implemented.'

        FLAGS.feature_extractor_img = FLAGS.img_clf_type

        # FLAGS.dir_clf += '_final'
        FLAGS.distributed = False
        run_training_procedure_clf(FLAGS)
        print('\n**********************\n\n')
