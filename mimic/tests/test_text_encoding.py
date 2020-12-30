# HK, 26.12.20

from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import create_dir_structure
from mimic.utils.filehandling import expand_paths, get_config_path
from mimic.utils.flags import parser
from mimic.utils.flags import setup_flags
from mimic.utils.flags import update_flags_with_config
from mimic.utils.text import tensor_to_text


TRUE_VAL = {
    'char': 'there is no focal consolidation, pleural effusion or pneumothorax.  bilateral nodular opacities that most likely represent nipple shadows. the cardiomediastinal silhouette is normal.  clips project over the left lung, potentially within the breast. the imaged upper abdomen is unremarkable. chronic deformity of the posterior left sixth and seventh ribs are noted.',
    'word': 'There is no focal consolidation , pleural effusion or pneumothorax . Bilateral nodular opacities that most likely represent nipple shadows . The cardiomediastinal silhouette is normal . Clips project over the left lung , potentially within the breast . The imaged upper abdomen is unremarkable . Chronic deformity of the posterior left sixth and seventh ribs are noted .'
}

def run_test_text_encoding(text_encoding: str):
    """
    Verify if text encoding works.
    """
    flags = parser.parse_args([])
    flags.config_path = get_config_path()
    flags = update_flags_with_config(flags.config_path, testing=True)
    flags = expand_paths(flags)
    flags.text_encoding = text_encoding
    flags.str_experiment = 'test'
    flags = create_dir_structure(flags, train=False)

    flags = setup_flags(flags, testing=True)

    mimic = MimicExperiment(flags)

    text_tensor_sample = mimic.dataset_train.__getitem__(0)[0]['text']
    one_hot = flags.text_encoding != 'word'
    str_joiner = '' if flags.text_encoding == 'char' else ' '
    text_sample = str_joiner.join(tensor_to_text(mimic, text_tensor_sample.unsqueeze(0), one_hot)[0])
    print(text_sample)
    print(mimic.dataset_train.report_findings[0])

    assert text_sample.startswith(TRUE_VAL[flags.text_encoding])


def test_char_encoding():
    run_test_text_encoding('char')


def test_word_encoding():
    run_test_text_encoding('word')


if __name__ == '__main__':
    # test_word_encoding()
    test_char_encoding()
