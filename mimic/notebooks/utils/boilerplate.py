import argparse
import json
import os
import warnings

import numpy as np
import torch
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mimic.dataio.MimicDataset import Mimic
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import expand_paths, get_config_path
from mimic.utils.flags import parser


def test_clfs(flags, img_size: int, text_encoding: str, alphabet=''):
    flags.img_size = img_size
    flags.text_encoding = text_encoding
    mimic_experiment = MimicExperiment(flags=flags)

    mimic_test = Mimic(flags, mimic_experiment.labels, split='eval')

    models = {}

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    results = {}
    for modality in ['PA', 'Lateral', 'text']:
        models[modality] = mimic_experiment.clfs[modality].eval()
        results[modality] = {
            'list_precision_vals': [],
            'list_prediction_vals': [],
            'list_gt_vals': [],
        }

    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels))))

        for modality in results:
            clf_input = Variable(batch_d[modality]).to(flags.device)
            prediction = models[modality](clf_input).cpu()
            results[modality]['list_prediction_vals'] = translate(prediction, results[modality]['list_prediction_vals'])
            results[modality]['list_gt_vals'] = translate(batch_l.cpu(), results[modality]['list_gt_vals'])
            prediction = prediction.data.numpy().ravel()
            avg_precision = average_precision_score(labels.ravel(), prediction)

            if not np.isnan(avg_precision):
                results[modality]['list_precision_vals'].append(avg_precision)
            else:
                warnings.warn(
                    f'avg_precision_{modality} has value {avg_precision} with labels: {labels.ravel()} and '
                    f'prediction: {prediction}')

    for modality in results:
        results[modality]['report'] = metrics.classification_report(results[modality]['list_gt_vals'],
                                                                    results[modality]['list_prediction_vals'], digits=4)
    return results


def translate(batch, list_labels: list) -> list:
    """
    Translates batch label tensor to list of character labels
    """
    for elem in batch:
        elem = elem.detach().numpy()
        if elem[0] == 1:
            list_labels.append('Lung Opacity')
        elif elem[1] == 1:
            list_labels.append('Pleural Effusion')
        elif elem[2] == 1:
            list_labels.append('Support Devices')
        else:
            list_labels.append('None')
    return list_labels


def test_clf(flags, mimic_experiment, mimic_test, modality: str):
    """
    Tests the classifier that the workflow would
    """
    model = mimic_experiment.clfs[modality]

    model.eval()

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    list_precision_vals = []
    list_prediction_vals = []
    list_labels = []
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]

        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels)))).ravel()

        clf_input = Variable(batch_d[modality]).to(flags.device)

        prediction = model(clf_input)
        list_prediction_vals = translate(prediction.cpu(), list_prediction_vals)
        list_labels = translate(batch_l.cpu(), list_labels)
        prediction = prediction.cpu().data.numpy().ravel()
        avg_precision = average_precision_score(labels, prediction)

        if not np.isnan(avg_precision):
            list_precision_vals.append(avg_precision)
        else:
            warnings.warn(
                f'avg_precision_{modality} has value {avg_precision} with labels: {labels.ravel()} and '
                f'prediction: {prediction.cpu().data.numpy().ravel()}')

    return metrics.classification_report(list_labels, list_prediction_vals, digits=4), list_precision_vals


def test_model(flags, model, clf_type: str, test_set, modality, labels):
    model.eval()

    dataloader = torch.utils.data.DataLoader(test_set, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    list_predicted_labels = []
    list_gt = []
    list_predictions = []
    list_labels = []
    for idx, (batch_d, batch_l) in enumerate(dataloader):
        if clf_type == 'densenet' and not modality == 'text':
            bs, n_crops, c, h, w = batch_d[modality].size()
            imgs = Variable(batch_d[modality].view(-1, c, h, w)).to(flags.device)
        else:
            imgs = Variable(batch_d[modality]).to(flags.device)
        gt = np.array(np.reshape(batch_l, (batch_l.shape[0], len(labels)))).ravel()

        prediction = model(imgs)
        if flags.img_clf_type == 'densenet' and modality != 'text':
            prediction = prediction.view(bs, n_crops, -1).mean(1)

        list_predicted_labels = translate(prediction.cpu(), list_predicted_labels)
        list_labels = translate(batch_l.cpu(), list_labels)
        prediction = prediction.cpu().data.numpy().ravel()
        list_gt.extend(gt)
        list_predictions.extend(prediction)

    avg_precision = average_precision_score(list_gt, list_predictions)

    return metrics.classification_report(list_labels, list_predicted_labels, digits=4), avg_precision


def test_dummy(flags, alphabet, modality: str = 'PA'):
    """
    Trains and evaluates a dummy classifier on the test set as baseline.
    Returns the average precision values
    """
    mimic_experiment = MimicExperiment(flags=flags)
    mimic_test = Mimic(flags, mimic_experiment.labels, split='eval')
    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=0,
                                             drop_last=True)
    list_batches = []
    list_labels = []
    list_precision_vals = []

    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        ground_truth = batch_l.cpu().data.numpy()
        clf_input = Variable(batch_d[modality]).cpu().data.numpy()
        list_batches.extend(clf_input)
        list_labels.extend(ground_truth)
    # dummy classifier has no partial_fit, so all the data must be fed at once
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(list_batches, list_labels)
    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        clf_input = Variable(batch_d[modality]).cpu().data.numpy()
        predictions = dummy_clf.predict(clf_input)
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels)))).ravel()
        avg_precision = average_precision_score(labels, predictions.ravel())
        if not np.isnan(avg_precision):
            list_precision_vals.append(avg_precision)
        else:
            warnings.warn(
                f'avg_precision_{modality} has value {avg_precision} with labels: {labels.ravel()} and '
                f'prediction: {predictions.cpu().data.numpy().ravel()}')
    return list_precision_vals


if __name__ == '__main__':
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    config_path = get_config_path()
    FLAGS = parser.parse_args()
    with open(config_path, 'rt') as json_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(json_file))
        FLAGS = parser.parse_args(namespace=t_args)
    FLAGS = expand_paths(FLAGS)
    print(FLAGS.dir_data)
    FLAGS.str_experiment = 'temp'
    FLAGS.device = 'cuda'
    FLAGS.dir_gen_eval_fid = ''
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]
    FLAGS.text_encoding = 'char'
    FLAGS.img_size = 128
    # MIMIC_EXPERIMENT = MimicExperiment(flags=FLAGS)
    # mimic_test = Mimic(FLAGS, MIMIC_EXPERIMENT.labels, split='eval')
    test_clfs(FLAGS, 256, 'text')
    asdasd = 0
    # results = test_clfs(FLAGS, 128, 'char', alphabet)

    # for modality in ['PA', 'Lateral', 'text']:
    #     report, list_precision = test_clf(FLAGS, MIMIC_EXPERIMENT, mimic_test, modality)

    # print(list_precision_pa)
    # print(list_precision_lat)
    # print(list_precision_text)
    # print(np.mean(list_precision_pa))
    # print(np.mean(list_precision_lat))
    # print(np.mean(list_precision_text))
