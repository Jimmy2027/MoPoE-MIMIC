from typing import Iterable
import torch
from mimic.utils.text import tensor_to_text


class TBLogger():
    def __init__(self, name, writer):
        self.name = name
        self.writer = writer
        self.training_prefix = 'train'
        self.testing_prefix = 'test'
        self.step = 0

    def write_log_probs(self, name, log_probs):
        self.writer.add_scalars('%s/LogProb' % name,
                                log_probs,
                                self.step)

    def write_klds(self, name, klds):
        self.writer.add_scalars('%s/KLD' % name,
                                klds,
                                self.step)

    def write_group_div(self, name, group_div):
        self.writer.add_scalars('%s/group_divergence' % name,
                                {'group_div': group_div},
                                self.step)

    def write_latent_distr(self, name, latents):
        for k, key in enumerate(latents.keys()):
            self.writer.add_scalars('%s/mu' % name,
                                    {key: latents[key][0]},
                                    self.step)
            self.writer.add_scalars('%s/logvar' % name,
                                    {key: latents[key][1]},
                                    self.step)

    def write_lr_eval(self, lr_eval):
        for l_key in sorted(lr_eval.keys()):
            mean_AP_keys = [k for k in lr_eval[l_key] if k.startswith('mean_AP')]
            results = {k: v for k, v in lr_eval[l_key].items() if k in ['dice', 'accuracy', *mean_AP_keys]}
            self.writer.add_scalars(f'Latent Representation/{l_key}', results, self.step)

    def write_coherence_logs(self, gen_eval):
        for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
            for k, s_key in enumerate(gen_eval['cond'][l_key].keys()):
                self.writer.add_scalars('Generation/%s/%s' %
                                        (l_key, s_key),
                                        gen_eval['cond'][l_key][s_key],
                                        self.step)
        self.writer.add_scalars('Generation/Random',
                                gen_eval['random'],
                                self.step)

    def write_lhood_logs(self, lhoods):
        for k, key in enumerate(sorted(lhoods.keys())):
            self.writer.add_scalars('Likelihoods/%s' %
                                    (key),
                                    lhoods[key],
                                    self.step)

    def write_prd_scores(self, prd_scores):
        self.writer.add_scalars('PRD',
                                prd_scores,
                                self.step)

    def write_plots(self, plots, epoch):
        for p_key in plots:
            ps = plots[p_key]
            for name in ps:
                fig = ps[name]
                self.writer.add_image(p_key + '_' + name,
                                      fig,
                                      epoch,
                                      dataformats="HWC")

    def add_basic_logs(self, name, joint_divergence, latents, loss, log_probs, klds):
        self.writer.add_scalars('%s/Loss' % name,
                                {'loss': loss},
                                self.step)
        self.write_log_probs(name, log_probs)
        self.write_klds(name, klds)
        self.write_group_div(name, joint_divergence)
        self.write_latent_distr(name, latents=latents)

    def write_training_logs(self, joint_divergence, latents, total_loss, log_probs, klds):
        self.add_basic_logs(self.training_prefix, joint_divergence, latents, total_loss, log_probs,
                            klds)

    def write_testing_logs(self, joint_divergence, latents, total_loss, log_probs, klds):
        self.add_basic_logs(self.testing_prefix, joint_divergence, latents, total_loss, log_probs, klds)

    def write_model_graph(self, model):
        """
        writes the model graph to tensorboard
        """
        self.writer.add_graph(model)

    def write_text(self, log_tag: str, text: str):
        self.writer.add_text(log_tag, text, global_step=self.step)

    def write_texts_from_list(self, log_tag: str, texts: Iterable[str], text_encoding: str):
        for i, text in enumerate(texts):
            sep = ' ' if text_encoding == 'word' else ''
            self.writer.add_text(log_tag, sep.join(text), global_step=i)

    def set_epoch(self, epoch: int):
        """
        Sets the epoch for all values that will be logged during that epoch.
        """
        self.step = epoch

    def write_tensor_to_text(self, text_tensor, exp, log_tag: str):
        sep = ' ' if exp.flags.text_encoding == 'word' else ''
        one_hot = exp.flags.text_encoding == 'char'
        self.writer.add_text(log_tag, sep.join(tensor_to_text(exp, text_tensor, one_hot=one_hot)),
                             global_step=self.step)
