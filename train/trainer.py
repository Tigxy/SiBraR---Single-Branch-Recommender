import os
import logging

from tqdm import trange, tqdm
import wandb
from ray.air import session

import torch
from torch.utils import data
from torch.profiler import profile, record_function, ProfilerActivity

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from data.config_classes import ExperimentConfig, EvalConfig
from eval.eval import evaluate_recommender_algorithm, FullEvaluator
from train.rec_losses import RecommenderSystemLoss


class Trainer:

    def __init__(self, model: SGDBasedRecommenderAlgorithm,
                 train_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 rec_loss: RecommenderSystemLoss,
                 conf: ExperimentConfig,
                 train_val_loader: data.DataLoader = None):
        """
        Train and Evaluate the model.
        :param model: Model to train
        :param train_loader: Training DataLoader
        :param val_loader: Validation DataLoader
        :param rec_loss: Recommendation Loss
        :param conf: Configuration dictionary
        """

        self.full_conf = conf
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_val_loader = train_val_loader
        self.evaluate_train_loader = self.train_val_loader is not None

        # do some error checking to ensure that we have all the data we need for later on
        if (train_val_loader is None) != (conf.train_eval is None):
            raise ValueError('Either both, a validation loader for the train set `train_val_loader` '
                             'and its validation configuration `conf.train_eval` must be specified, or neither one!')

        self.profile_training = conf.profile_training
        self.profiling_count = 0

        self.learning_config = conf.learn

        self.device = conf.run_settings.device

        self.model = model
        self.pointer_to_model = self.model
        self.model.to(self.device)

        self.rec_loss = rec_loss

        self.lr = self.learning_config.lr
        self.wd = self.learning_config.wd

        opt_map = {
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
            'adamw': torch.optim.AdamW
        }
        self.optimizer = opt_map[self.learning_config.optimizer](self.model.parameters(),
                                                                 lr=self.lr, weight_decay=self.wd)

        self.n_epochs = self.learning_config.n_epochs
        self.optimizing_metric = self.learning_config.optimizing_metric
        self.max_patience = self.learning_config.max_patience

        self.model_path = conf.results_path

        self.use_wandb = conf.wandb.use_wandb
        self.batch_verbose = conf.run_settings.batch_verbose

        self._in_tune = conf.run_settings.in_tune

        self.best_value = None
        self.best_metrics = None
        self.best_epoch = None
        logging.info(f'Built Trainer module \n'
                     f'- n_epochs: {self.n_epochs} \n'
                     f'- rec_loss: {self.rec_loss.__class__.__name__} \n'
                     f'- loss_aggregator: {self.rec_loss.aggregator} \n'
                     f'- device: {self.device} \n'
                     f'- optimizing_metric: {self.optimizing_metric} \n'
                     f'- model_path: {self.model_path} \n'
                     f'- optimizer: {self.optimizer.__class__.__name__} \n'
                     f'- lr: {self.lr} \n'
                     f'- wd: {self.wd} \n'
                     f'- use_wandb: {self.use_wandb} \n'
                     f'- batch_verbose: {self.batch_verbose} \n'
                     f'- max_patience: {self.max_patience} \n')

    def fit(self):
        """
        Runs the Training procedure
        """
        current_patience = self.max_patience
        log_dict = self.val()

        self.best_value = log_dict['max_optimizing_metric'] = log_dict[self.optimizing_metric]
        self.best_epoch = log_dict['best_epoch'] = -1
        self.best_metrics = log_dict
        if hasattr(self.pointer_to_model, 'post_val') and callable(self.pointer_to_model.post_val):
            log_dict.update(self.pointer_to_model.post_val(-1))

        print(f'Init - {self.optimizing_metric}={self.best_value:.4f}')

        if self.use_wandb and not self._in_tune:
            wandb.log(log_dict)

        if self._in_tune:
            session.report(log_dict)

        self.pointer_to_model.save_model_to_path(self.model_path)

        for epoch in trange(self.n_epochs, desc='epochs'):

            self.model.train()

            if current_patience == 0:
                print('Ran out of patience, stopping ')
                break

            epoch_losses = self.train()

            epoch_str = f'Epoch [{epoch:>3d}|{self.n_epochs:>d}]'
            print(f'\n{epoch_str} - average train loss {epoch_losses["train/loss"]:.4f} '
                  f'({epoch_losses["train/rec_loss"]:.4f} recommendation loss '
                  f'+ {epoch_losses["train/reg_loss"]:.4f} regularization loss)')

            if self.evaluate_train_loader:
                epoch_losses.update(**self.train_val())

            metrics_values = self.val()

            curr_value = metrics_values[self.optimizing_metric]
            result_str = f'{self.optimizing_metric}={curr_value:.4f}'
            print(f'\n{epoch_str} - validation {result_str}')

            if curr_value > self.best_value:
                self.best_value = metrics_values['max_optimizing_metric'] = curr_value
                self.best_epoch = metrics_values['best_epoch'] = epoch
                self.best_metrics = metrics_values

                print(f'{epoch_str} - new best model found (validation {result_str})\n')
                self.pointer_to_model.save_model_to_path(self.model_path)

                current_patience = self.max_patience  # Reset patience
            else:
                metrics_values['max_optimizing_metric'] = self.best_value
                current_patience -= 1

            # Logging
            log_dict = {**metrics_values, **epoch_losses}
            # Execute a post validation function that is specific to the model
            if hasattr(self.pointer_to_model, 'post_val') and callable(self.pointer_to_model.post_val):
                log_dict.update(self.pointer_to_model.post_val(epoch))

            if self.use_wandb and not self._in_tune:
                wandb.log(log_dict)

            if self._in_tune:
                session.report(log_dict)

        return self.best_metrics

    def train(self):
        if self.profile_training:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function("training"):
                    epoch_losses = self._train()

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            trace_path = os.path.join(self.full_conf.results_path, f'trace_{self.profiling_count}.json')
            print(f'storing full trace to "{trace_path}"')
            prof.export_chrome_trace(trace_path)
            self.profiling_count += 1
        else:
            epoch_losses = self._train()
        return epoch_losses

    def _train(self):

        # prevent training model in eval mode
        self.model.train()

        epoch_losses = {
            'loss': 0,
            'rec_loss': 0,
        }

        desc = 'training'
        if self.batch_verbose:
            iterator = tqdm(self.train_loader, desc=desc)
        else:
            print(f'{desc}...')
            iterator = self.train_loader

        for batch_count, (u_idxs, i_idxs, labels) in enumerate(iterator):
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            rec_loss = self.rec_loss.compute_loss(out, labels)
            reg_losses = self.pointer_to_model.get_and_reset_other_loss()
            reg_loss = reg_losses['reg_loss'].to(rec_loss.device)

            total_loss = rec_loss + reg_loss

            epoch_losses['loss'] += total_loss.item()
            epoch_losses['rec_loss'] += rec_loss.item()
            epoch_losses.update({k: reg_losses[k].item() + epoch_losses.get(k, 0) for k in reg_losses})

            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            max_batches = self.full_conf.learn.max_batches_per_epoch
            if max_batches is not None and max_batches <= batch_count + 1:  # +1 because iterator starts at 0
                print(f'limit of {max_batches} batches hit, thus stopping this training cycle.')
                break

            if self.profile_training and batch_count >= 50:
                break

        epoch_losses = {f'train/{k}': v / len(self.train_loader) for k, v in epoch_losses.items()}
        return epoch_losses

    @torch.no_grad()
    def _eval_loader(self, loader: data.DataLoader, config: EvalConfig, evaluator_name: str = None):
        """
        Runs the evaluation procedure.
        :return: the dictionary of the metric values
        """
        self.model.eval()
        evaluator = FullEvaluator(config=config, evaluator_name=evaluator_name, dataset=loader.dataset)
        metrics_values = evaluate_recommender_algorithm(self.pointer_to_model, loader, evaluator, self.device,
                                                        verbose=self.batch_verbose)
        return metrics_values

    def train_val(self):
        return self._eval_loader(self.train_val_loader, self.full_conf.train_eval, 'train')

    def val(self):
        """
        Runs the evaluation procedure.
        :return: the dictionary of the metric values
        """
        return self._eval_loader(self.val_loader, self.full_conf.eval)
