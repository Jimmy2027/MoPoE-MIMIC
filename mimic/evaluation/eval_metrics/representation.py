import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimic.networks.VAEtrimodalMimic import VAEtrimodalMimic
from mimic.utils.experiment import MimicExperiment


def train_clf_lr_all_subsets(exp: MimicExperiment):
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    mm_vae: VAEtrimodalMimic
    subsets = exp.subsets

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=args.dataloader_workers // args.world_size if args.distributed
                          else args.dataloader_workers,
                          drop_last=True)
    if exp.flags.steps_per_training_epoch > 0:
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(d_loader)

    bs = exp.flags.batch_size
    class_dim = exp.flags.class_dim
    n_samples = int(exp.dataset_train.__len__())
    data_train = {
        s_key: np.zeros((n_samples, class_dim))
        for k, s_key in enumerate(subsets.keys())
        if s_key != ''
    }

    all_labels = np.zeros((n_samples, len(exp.labels)))
    for it, (batch_d, batch_l) in tqdm(enumerate(d_loader), total=training_steps, postfix='train_clf_lr'):
        """
        Constructs the training set (labels and inferred subsets) for the classifier training.
        """
        if it > training_steps and len(np.unique(all_labels)) > 1:
            # labels need at least 2 classes to train the clf
            break

        batch_d = {k: v.to(exp.flags.device) for k, v in batch_d.items()}
        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)

        lr_subsets = inferred['subsets']
        all_labels[(it * bs):((it + 1) * bs), :] = np.reshape(batch_l, (bs,
                                                                        len(exp.labels)))
        for k, key in enumerate(lr_subsets.keys()):
            data_train[key][(it * bs):((it + 1) * bs), :] = lr_subsets[key][0].cpu().data.numpy()

    n_train_samples = exp.flags.num_training_samples_lr
    # get random labels such that it contains both classes
    labels, rand_ind_train = get_random_labels(n_samples, n_train_samples, all_labels)
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            d = data_train[s_key]
            data_train[s_key] = d[rand_ind_train, :]
    return train_clf_lr(exp, data_train, labels)


def get_random_labels(n_samples, n_train_samples, all_labels, max_tries=1000):
    """
    The classifier needs labels from both classes to train. This function resamples "all_labels"
    until it contains examples from both classes
    """
    assert any(len(np.unique(all_labels[:, l])) > 1 for l in range(all_labels.shape[-1])), \
        'The labels must contain at least two classes to train the classifier'
    rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
    labels = all_labels[rand_ind_train, :]
    tries = 1
    while any(len(np.unique(labels[:, l])) <= 1 for l in range(labels.shape[-1])):
        rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
        labels = all_labels[rand_ind_train, :]
        tries += 1
        assert max_tries >= tries, f'Could not get sample containing both classes to train ' \
                                   f'the classifier in {tries} tries. Might need to increase batch_size'
    return labels, rand_ind_train


def test_clf_lr_all_subsets(epoch, clf_lr, exp):
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    subsets = exp.subsets

    lr_eval = {
        label_str: {
            s_key: [] for k, s_key in enumerate(subsets.keys()) if s_key != ''
        }
        for l, label_str in enumerate(exp.labels)
    }

    d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True)

    if exp.flags.steps_per_training_epoch > 0:
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(d_loader)

    for iteration, batch in enumerate(d_loader):
        if iteration > training_steps:
            break
        batch_d = batch[0]
        batch_l = batch[1]
        batch_d = {k: v.to(exp.flags.device) for k, v in batch_d.items()}

        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        lr_subsets = inferred['subsets']
        data_test = {
            key: lr_subsets[key][0].cpu().data.numpy()
            for k, key in enumerate(lr_subsets.keys())
        }

        evals = classify_latent_representations(exp,
                                                epoch,
                                                clf_lr,
                                                data_test,
                                                batch_l)
        for l, label_str in enumerate(exp.labels):
            eval_label = evals[label_str]
            for k, s_key in enumerate(eval_label.keys()):
                lr_eval[label_str][s_key].append(eval_label[s_key])
    for l, l_key in enumerate(lr_eval.keys()):
        lr_eval_label = lr_eval[l_key]
        for k, s_key in enumerate(lr_eval_label.keys()):
            lr_eval[l_key][s_key] = exp.mean_eval_metric(lr_eval_label[s_key])
    return lr_eval


def classify_latent_representations(exp, epoch, clf_lr, data, labels):
    labels = np.array(np.reshape(labels, (labels.shape[0], len(exp.labels))))
    eval_all_labels = {}
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l]
        clf_lr_label = clf_lr[label_str]
        eval_all_reps = {}
        for s_key in data.keys():
            data_rep = data[s_key]
            clf_lr_rep = clf_lr_label[s_key]
            if exp.flags.dataset == 'testing':
                # when using the testing dataset, the vae data_rep might contain nans. Replace them for testing purposes
                y_pred_rep = clf_lr_rep.predict(np.nan_to_num(data_rep))
            else:
                y_pred_rep = clf_lr_rep.predict(data_rep)
            eval_label_rep = exp.eval_metric(gt.ravel(),
                                             y_pred_rep.ravel())
            eval_all_reps[s_key] = eval_label_rep
        eval_all_labels[label_str] = eval_all_reps
    return eval_all_labels


def train_clf_lr(exp, data, labels):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)))
    clf_lr_labels = {}
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l]
        clf_lr_reps = {}
        for s_key in data.keys():
            data_rep = data[s_key]
            clf_lr_s = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
            if exp.flags.dataset == 'testing':
                # when using the testing dataset, the vae data_rep might contain nans. Replace them for testing purposes
                clf_lr_s.fit(np.nan_to_num(data_rep), gt.ravel())
            else:
                clf_lr_s.fit(data_rep, gt.ravel())
            clf_lr_reps[s_key] = clf_lr_s
        clf_lr_labels[label_str] = clf_lr_reps
    return clf_lr_labels
