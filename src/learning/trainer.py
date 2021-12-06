"""
Train the passed model given the data and the batcher and save the best to disk.
"""
from __future__ import print_function
import os
import logging
import time, copy
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import transformers

from . import predict_utils as pu
from . import data_utils as du


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    r"""Strip the prefix in state_dict, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    Copied from here cause im using version 1.8.1 and this is in 1.9.0
    https://github.com/pytorch/pytorch/blob/1f2b96e7c447210072fe4d2ed1a39d6121031ba6/torch/nn/modules/utils.py
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)
    
    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.
            
            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def generic_save_function_ddp(model, save_path, model_suffix):
    """
    Model saving function used in the training loop.
    This is saving with the assumption that non DDP models will use this model.
    """
    model_fname = os.path.join(save_path, f'model_{model_suffix}.pt')
    model = copy.deepcopy(model.state_dict())
    consume_prefix_in_state_dict_if_present(model, "module.")
    torch.save(model, model_fname)
    print('Wrote: {:s}'.format(model_fname))


def generic_save_function(model, save_path, model_suffix):
    """
    Model saving function used in the training loop.
    """
    model_fname = os.path.join(save_path, f'model_{model_suffix}.pt')
    torch.save(model.state_dict(), model_fname)
    logging.info('Wrote: {:s}'.format(model_fname))


def sentbert_save_function(model, save_path, model_suffix):
    """
    Model saving function used in the training loop for sentence bert
    """
    model_fname = os.path.join(save_path, f'sent_encoder_{model_suffix}.pt')
    torch.save(model.sent_encoder.state_dict(), model_fname)
    logging.info('Wrote: {:s}'.format(model_fname))


def ictbert_save_function(model, save_path, model_suffix):
    """
    Model saving function used in the training loop for sentence bert
    """
    model_fname = os.path.join(save_path, f'sent_encoder_{model_suffix}.pt')
    torch.save(model.sent_encoder.state_dict(), model_fname)
    logging.info('Wrote: {:s}'.format(model_fname))
    model_fname = os.path.join(save_path, f'context_encoder_{model_suffix}.pt')
    torch.save(model.context_encoder.state_dict(), model_fname)
    logging.info('Wrote: {:s}'.format(model_fname))


class GenericTrainer:
    # If this isnt set outside of here it crashes with: "got multiple values for argument"
    # todo: Look into who this happens --low-pri.s
    save_function = generic_save_function
    
    def __init__(self, cml_exp, model, batcher, model_path, train_hparams,
                 early_stop=True, verbose=True, dev_score='loss'):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other models should subclass this and define the data that the models
        being trained consume.
        :param cml_exp: comet_ml Experiment object.
        :param model: pytorch model.
        :param batcher: a model_utils.Batcher class.
        :param model_path: string; directory to which model should get saved.
        :param early_stop: boolean;
        :param verbose: boolean;
        :param dev_score: string; {'loss'/'f1'} How dev set evaluation should be done.
        # train_hparams dict elements.
        :param train_size: int; number of training examples.
        :param dev_size: int; number of dev examples.
        :param batch_size: int; number of examples per batch.
        :param accumulated_batch_size: int; number of examples to accumulate gradients
            for in smaller batch size before computing the gradient. If this is not present
            in the dictionary or is smaller than batch_size then assume no gradient
            accumulation.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param es_check_every: int; check some metric on the dev set every check_every iterations.
        :param lr_decay_method: string; {'exponential', 'warmuplin', 'warmupcosine'}
        :param decay_lr_by: float; decay the learning rate exponentially by the following
            factor.
        :param num_warmup_steps: int; number of steps for which to do warm up.
        :param decay_lr_every: int; decay learning rate every few iterations.
        
        """
        # Book keeping
        self.cml_experiment = cml_exp
        self.dev_score = dev_score
        self.verbose = verbose
        self.es_check_every = train_hparams['es_check_every']
        self.num_train = train_hparams['train_size']
        self.num_dev = train_hparams['dev_size']
        self.batch_size = train_hparams['batch_size']
        self.num_epochs = train_hparams['num_epochs']
        try:
            self.accumulated_batch_size = train_hparams['accumulated_batch_size']
            # You can set accumulated_batch_size to 0 or -1 and it will assume no grad accumulation.
            if self.accumulated_batch_size > 0:
                # It should be bigger and an exact multiple of the batch size.
                assert(self.accumulated_batch_size > self.batch_size
                       and self.accumulated_batch_size % self.batch_size == 0)
                self.accumulate_gradients = True
                self.update_params_every = self.accumulated_batch_size/self.batch_size
                logging.info('Accumulating gradients for: {:}; updating params every: {:}; with batch size: {:}'
                             .format(self.accumulated_batch_size, self.update_params_every, self.batch_size))
            else:
                self.accumulate_gradients = False
        except KeyError:
            self.accumulate_gradients = False
        if self.num_train > self.batch_size:
            self.num_batches = int(np.ceil(float(self.num_train)/self.batch_size))
        else:
            self.num_batches = 1
        self.model_path = model_path  # Save model and checkpoints.
        self.total_iters = self.num_epochs*self.num_batches
        self.iteration = 0

        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.time_per_batch = 0
        self.time_per_dev_pass = 0
        # Different trainer classes can add this based on the data that the model
        # they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}

        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = train_hparams['update_rule']
        self.learning_rate = train_hparams['learning_rate']

        # Initialize optimizer.
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.update_rule == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Unknown upate rule: {:s}'.format(self.update_rule))
        # Reduce the learning rate every few iterations.
        self.lr_decay_method = train_hparams['lr_decay_method']
        self.decay_lr_every = train_hparams['decay_lr_every']
        self.log_every = 5
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.1, patience=1,
        #     verbose=True)
        if self.lr_decay_method == 'exponential':
            self.decay_lr_by = train_hparams['decay_lr_by']
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                              gamma=self.decay_lr_by)
        elif self.lr_decay_method == 'warmuplin':
            self.num_warmup_steps = train_hparams['num_warmup_steps']
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps,
                # Total number of training batches.
                num_training_steps=self.num_epochs*self.num_batches)
        elif self.lr_decay_method == 'warmupcosine':
            self.num_warmup_steps = train_hparams['num_warmup_steps']
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_epochs*self.num_batches)
        else:
            raise ValueError('Unknown lr_decay_method: {:}'.format(train_hparams['lr_decay_method']))

        # Train statistics.
        self.loss_history = defaultdict(list)
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []
        
        # Every subclass needs to set this.
        self.loss_function_cal = GenericTrainer.compute_loss

    def train(self):
        """
        Make num_epoch passes through the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_score = -np.inf

        total_time_per_batch = 0
        total_time_per_dev = 0
        train_start = time.time()
        logging.info('num_train: {:d}; num_dev: {:d}'.format(
            self.num_train, self.num_dev))
        logging.info('Training {:d} epochs, {:d} iterations'.
                     format(self.num_epochs, self.total_iters))
        with self.cml_experiment.train():
            for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
                # Initialize batcher. Shuffle one time before the start of every
                # epoch.
                epoch_batcher = self.batcher(ex_fnames=ex_fnames,
                                             num_examples=self.num_train,
                                             batch_size=self.batch_size)
                # Get the next training batch.
                iters_start = time.time()
                for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
                    self.model.train()
                    batch_start = time.time()
                    # Impelemented according to:
                    # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-
                    # manually-to-zero-in-pytorch/4903/20
                    # With my implementation a final batch update may not happen sometimes but shrug.
                    if self.accumulate_gradients:
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters.
                        objective.backward()
                        if (self.iteration + 1) % self.update_params_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    else:
                        # Clear all gradient buffers.
                        self.optimizer.zero_grad()
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters.
                        objective.backward()
                        # Step in the direction of the gradient.
                        self.optimizer.step()
                    if self.iteration % self.log_every == 0:
                        # Save every loss component separately.
                        loss_str = []
                        for key in ret_dict:
                            if torch.cuda.is_available():
                                loss_comp = float(ret_dict[key].data.cpu().numpy())
                            else:
                                loss_comp = float(ret_dict[key].data.numpy())
                            self.loss_history[key].append(loss_comp)
                            loss_str.append('{:s}: {:.4f}'.format(key, loss_comp))
                        self.loss_checked_iters.append(self.iteration)
                        if self.verbose:
                            log_str = 'Epoch: {:d}; Iteration: {:d}/{:d}; '.format(epoch, self.iteration, self.total_iters)
                            logging.info(log_str + '; '.join(loss_str))
                    elif self.verbose:
                        logging.info('Epoch: {:d}; Iteration: {:d}/{:d}'.
                                     format(epoch, self.iteration, self.total_iters))
                    # The decay_lr_every doesnt need to be a multiple of self.log_every
                    if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                        self.scheduler.step()
                        # logging.info('Decayed learning rates: {}'.
                        #              format([g['lr'] for g in self.optimizer.param_groups]))
                    batch_end = time.time()
                    total_time_per_batch += batch_end-batch_start
                    # Check every few iterations how you're doing on the dev set.
                    with self.cml_experiment.validate():
                        if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop:
                            # Save the loss at this point too.
                            for key in ret_dict:
                                if torch.cuda.is_available():
                                    loss_comp = float(ret_dict[key].data.cpu().numpy())
                                else:
                                    loss_comp = float(ret_dict[key].data.numpy())
                                self.loss_history[key].append(loss_comp)
                            self.loss_checked_iters.append(self.iteration)
                            # Switch to eval model and check loss on dev set.
                            self.model.eval()
                            dev_start = time.time()
                            # Returns the dev F1.
                            if self.dev_score == 'f1':
                                dev_score = pu.batched_dev_scores(
                                    model=self.model, batcher=self.batcher, batch_size=self.batch_size,
                                    ex_fnames=self.dev_fnames, num_examples=self.num_dev)
                            elif self.dev_score == 'loss':
                                dev_score = -1.0 * pu.batched_loss(
                                    model=self.model, batcher=self.batcher, batch_size=self.batch_size,
                                    ex_fnames=self.dev_fnames, num_examples=self.num_dev,
                                    loss_helper=self.loss_function_cal)
                            self.cml_experiment.log_metric(self.dev_score, dev_score, step=self.iteration)
                            dev_end = time.time()
                            total_time_per_dev += dev_end-dev_start
                            self.dev_score_history.append(dev_score)
                            self.dev_checked_iters.append(self.iteration)
                            if dev_score > best_dev_score:
                                best_dev_score = dev_score
                                # Deep copy so you're not just getting a reference.
                                best_params = copy.deepcopy(self.model.state_dict())
                                best_epoch = epoch
                                best_iter = self.iteration
                                everything = (epoch, self.iteration, self.total_iters, dev_score)
                                if self.verbose:
                                    logging.info('Current best model; Epoch {:d}; '
                                                 'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                                self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')
                            else:
                                everything = (epoch, self.iteration, self.total_iters, dev_score)
                                if self.verbose:
                                    logging.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                    self.iteration += 1
                epoch_time = time.time()-iters_start
                logging.info('Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
                logging.info('\n')
            
        # Say how long things took.
        train_time = time.time()-train_start
        logging.info('Training time: {:.4f}s'.format(train_time))
        if self.total_iters > 0:
            self.time_per_batch = float(total_time_per_batch)/self.total_iters
        else:
            self.time_per_batch = 0.0
        logging.info('Time per batch: {:.4f}s'.format(self.time_per_batch))
        if self.early_stop and self.dev_score_history:
            if len(self.dev_score_history) > 0:
                self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history)
            else:
                self.time_per_dev_pass = 0
            logging.info('Time per dev pass: {:4f}s'.format(self.time_per_dev_pass))

        # Save the learnt model: save both the final model and the best model.
        # https://stackoverflow.com/a/43819235/3262406
        self.save_function(model=self.model, save_path=self.model_path, model_suffix='final')
        logging.info('Best model; Epoch {:d}; Iteration {:d}; Dev loss: {:.4f}'
                     .format(best_epoch, best_iter, best_dev_score))
        # self.model.load_state_dict(best_params)
        # self.save_function(model=self.model, save_path=self.model_path, model_suffix='best')

        # Plot training time stats.
        for key in self.loss_history:
            du.plot_train_hist(self.loss_history[key], self.loss_checked_iters,
                               fig_path=self.model_path, ylabel=key)
        du.plot_train_hist(self.dev_score_history, self.dev_checked_iters,
                           fig_path=self.model_path, ylabel='Dev-set Score')
        
    @staticmethod
    def compute_loss(loss_components):
        """
        Models will return dict with different loss components, use this and compute batch loss.
        :param loss_components: dict('str': Variable)
        :return:
        """
        raise NotImplementedError


class BasicTrainer(GenericTrainer):
    def __init__(self, model, data_path, batcher, train_size, dev_size,
                 batch_size, update_rule, num_epochs, learning_rate,
                 check_every, decay_lr_by, decay_lr_every, model_path, early_stop=True,
                 verbose=True, dev_score='f1'):
        """
        Trainer for any model returning NLL. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        # Todo: Change trainer API.
        raise NotImplementedError
        GenericTrainer.__init__(self, model, batcher, train_size, dev_size,
                                batch_size, update_rule, num_epochs, learning_rate,
                                check_every, decay_lr_by, decay_lr_every, model_path,
                                early_stop, verbose, dev_score)
        # Expect the presence of a directory with as many shuffled copies of the
        # dataset as there are epochs and a negative examples file.
        self.train_fnames = []
        for i in range(self.num_epochs):
            ex_fname = {
                'pos_ex_fname': os.path.join(data_path, 'shuffled_data', 'train-{:d}.jsonl'.format(i)),
            }
            self.train_fnames.append(ex_fname)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, 'dev.jsonl'),
        }
        # Every subclass needs to set this.
        self.loss_function_cal = BasicTrainer.compute_loss
    
    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('nll': data likelihood)
        :return: Variable.
        """
        return loss_components['nll']


class BasicRankingTrainer(GenericTrainer):
    def __init__(self, cml_exp, model, batcher, model_path, data_path,
                 train_hparams, early_stop=True, verbose=True, dev_score='loss'):
        """
        Trainer for any model returning a ranking loss. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainer.__init__(self, cml_exp=cml_exp, model=model, batcher=batcher, model_path=model_path,
                                train_hparams=train_hparams, early_stop=early_stop, verbose=verbose,
                                dev_score=dev_score)
        # Expect the presence of a directory with as many shuffled copies of the
        # dataset as there are epochs and a negative examples file.
        self.train_fnames = []
        # Expect these to be there for the case of using diff kinds of training data for the same
        # model; hard negatives models, different alignment models and so on.
        if 'train_suffix' in train_hparams:
            suffix = train_hparams['train_suffix']
            train_basename = 'train-{:s}'.format(suffix)
            dev_basename = 'dev-{:s}'.format(suffix)
        else:
            train_basename = 'train'
            dev_basename = 'dev'
        for i in range(self.num_epochs):
            # Each run contains a copy of shuffled data for itself.
            ex_fname = {
                'pos_ex_fname': os.path.join(model_path, 'shuffled_data', '{:s}-{:d}.jsonl'.format(train_basename, i)),
            }
            self.train_fnames.append(ex_fname)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, '{:s}.jsonl'.format(dev_basename)),
        }
        # Every subclass needs to set this.
        self.loss_function_cal = BasicRankingTrainer.compute_loss
    
    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('rankl': rank loss value)
        :return: Variable.
        """
        return loss_components['rankl']


def conditional_log(logger, process_rank, message):
    """
    Helper to log only from one process when using DDP.
    -- logger is entirely unused. Dint seem to work when used in conjuncton with cometml.
    """
    if process_rank == 0:
        print(message)


class GenericTrainerDDP:
    # If this isnt set outside of here it crashes with: "got multiple values for argument"
    # todo: Look into who this happens --low-pri.s
    save_function = generic_save_function_ddp
    
    def __init__(self, cml_exp, logger, process_rank, num_gpus, model, batcher, model_path, train_hparams,
                 early_stop=True, verbose=True, dev_score='loss'):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other models should subclass this and define the data that the models
        being trained consume.
        :param cml_exp: comet_ml Experiment object.
        :param logger: a logger to write logs with.
        :param process_rank: int; which process this is.
        :param num_gpus: int; how many gpus are being used to train.
        :param model: pytorch model.
        :param batcher: a model_utils.Batcher class.
        :param model_path: string; directory to which model should get saved.
        :param early_stop: boolean;
        :param verbose: boolean;
        :param dev_score: string; {'loss'/'f1'} How dev set evaluation should be done.
        # train_hparams dict elements.
        :param train_size: int; number of training examples.
        :param dev_size: int; number of dev examples.
        :param batch_size: int; number of examples per batch.
        :param accumulated_batch_size: int; number of examples to accumulate gradients
            for in smaller batch size before computing the gradient. If this is not present
            in the dictionary or is smaller than batch_size then assume no gradient
            accumulation.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param es_check_every: int; check some metric on the dev set every check_every iterations.
        :param lr_decay_method: string; {'exponential', 'warmuplin', 'warmupcosine'}
        :param decay_lr_by: float; decay the learning rate exponentially by the following
            factor.
        :param num_warmup_steps: int; number of steps for which to do warm up.
        :param decay_lr_every: int; decay learning rate every few iterations.
        
        """
        # Book keeping
        self.process_rank = process_rank
        self.logger = logger
        self.cml_experiment = cml_exp
        self.dev_score = dev_score
        self.verbose = verbose
        self.es_check_every = train_hparams['es_check_every']//num_gpus
        self.num_dev = train_hparams['dev_size']
        self.batch_size = train_hparams['batch_size']
        self.num_epochs = train_hparams['num_epochs']
        try:
            self.accumulated_batch_size = train_hparams['accumulated_batch_size']
            # You can set accumulated_batch_size to 0 or -1 and it will assume no grad accumulation.
            if self.accumulated_batch_size > 0:
                # It should be bigger and an exact multiple of the batch size.
                assert(self.accumulated_batch_size > self.batch_size
                       and self.accumulated_batch_size % self.batch_size == 0)
                self.accumulate_gradients = True
                self.update_params_every = self.accumulated_batch_size/self.batch_size
                conditional_log(logger, process_rank,
                                'Accumulating gradients for: {:}; updating params every: {:}; with batch size: {:}'
                                .format(self.accumulated_batch_size, self.update_params_every, self.batch_size))
            else:
                self.accumulate_gradients = False
        except KeyError:
            self.accumulate_gradients = False
        self.model_path = model_path  # Save model and checkpoints.
        self.iteration = 0
        
        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.time_per_batch = 0
        self.time_per_dev_pass = 0
        # Different trainer classes can add this based on the data that the model
        # they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}
        
        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = train_hparams['update_rule']
        self.learning_rate = train_hparams['learning_rate']
        
        # Initialize optimizer.
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.update_rule == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Unknown upate rule: {:s}'.format(self.update_rule))
        # Reduce the learning rate every few iterations.
        self.lr_decay_method = train_hparams['lr_decay_method']
        self.decay_lr_every = train_hparams['decay_lr_every']
        self.log_every = 5
       
        # Train statistics.
        self.loss_history = defaultdict(list)
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []
        
        # Every subclass needs to set this.
        self.loss_function_cal = GenericTrainer.compute_loss
    
    def train(self):
        """
        Make num_epoch passes through the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_score = -np.inf
        
        total_time_per_batch = 0
        total_time_per_dev = 0
        train_start = time.time()
        conditional_log(self.logger, self.process_rank,
                        f'num_train: {self.num_train}; num_dev: {self.num_dev}')
        conditional_log(self.logger, self.process_rank,
                        f'Training {self.num_epochs} epochs, {self.total_iters} iterations')
        with self.cml_experiment.train():
            for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
                # Initialize batcher. Shuffle one time before the start of every
                # epoch.
                epoch_batcher = self.batcher(ex_fnames=ex_fnames,
                                             num_examples=self.num_train,
                                             batch_size=self.batch_size)
                # Get the next training batch.
                iters_start = time.time()
                for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
                    self.model.train()
                    batch_start = time.time()
                    # Impelemented according to:
                    # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-
                    # manually-to-zero-in-pytorch/4903/20
                    # With my implementation a final batch update may not happen sometimes but shrug.
                    if self.accumulate_gradients:
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters.
                        objective.backward()
                        if (self.iteration + 1) % self.update_params_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    else:
                        # Clear all gradient buffers.
                        self.optimizer.zero_grad()
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters.
                        objective.backward()
                        # Step in the direction of the gradient.
                        self.optimizer.step()
                    if self.iteration % self.log_every == 0:
                        # Save every loss component separately.
                        loss_str = []
                        for key in ret_dict:
                            if torch.cuda.is_available():
                                loss_comp = float(ret_dict[key].data.cpu().numpy())
                            else:
                                loss_comp = float(ret_dict[key].data.numpy())
                            self.loss_history[key].append(loss_comp)
                            loss_str.append('{:s}: {:.4f}'.format(key, loss_comp))
                        self.loss_checked_iters.append(self.iteration)
                        if self.verbose:
                            log_str = 'Epoch: {:d}; Iteration: {:d}/{:d}; '.format(epoch, self.iteration, self.total_iters)
                            conditional_log(self.logger, self.process_rank, log_str + '; '.join(loss_str))
                    elif self.verbose:
                        conditional_log(self.logger, self.process_rank,
                                        f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}')
                    # The decay_lr_every doesnt need to be a multiple of self.log_every
                    if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                        self.scheduler.step()
                    batch_end = time.time()
                    total_time_per_batch += batch_end-batch_start
                    # Check every few iterations how you're doing on the dev set.
                    if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop\
                            and self.process_rank == 0:
                        with self.cml_experiment.validate():
                            # Save the loss at this point too.
                            for key in ret_dict:
                                if torch.cuda.is_available():
                                    loss_comp = float(ret_dict[key].data.cpu().numpy())
                                else:
                                    loss_comp = float(ret_dict[key].data.numpy())
                                self.loss_history[key].append(loss_comp)
                            self.loss_checked_iters.append(self.iteration)
                            # Switch to eval model and check loss on dev set.
                            self.model.eval()
                            dev_start = time.time()
                            # Using the module as it is for eval:
                            # https://discuss.pytorch.org/t/distributeddataparallel-
                            # barrier-doesnt-work-as-expected-during-evaluation/99867/11
                            dev_score = -1.0 * pu.batched_loss_ddp(
                                model=self.model.module, batcher=self.batcher, batch_size=self.batch_size,
                                ex_fnames=self.dev_fnames, num_examples=self.num_dev,
                                loss_helper=self.loss_function_cal, logger=self.logger)
                            self.cml_experiment.log_metric(self.dev_score, dev_score, step=self.iteration)
                            dev_end = time.time()
                            total_time_per_dev += dev_end-dev_start
                            self.dev_score_history.append(dev_score)
                            self.dev_checked_iters.append(self.iteration)
                            if dev_score > best_dev_score:
                                best_dev_score = dev_score
                                # Deep copy so you're not just getting a reference.
                                best_params = copy.deepcopy(self.model.state_dict())
                                best_epoch = epoch
                                best_iter = self.iteration
                                everything = (epoch, self.iteration, self.total_iters, dev_score)
                                if self.verbose:
                                    self.logger.info('Current best model; Epoch {:d}; '
                                                     'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                                self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')
                            else:
                                everything = (epoch, self.iteration, self.total_iters, dev_score)
                                if self.verbose:
                                    self.logger.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'
                                                     .format(*everything))
                    dist.barrier()
                    self.iteration += 1
                epoch_time = time.time()-iters_start
                conditional_log(self.logger, self.process_rank, 'Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
                conditional_log(self.logger, self.process_rank, '\n')
        
        # Say how long things took.
        train_time = time.time()-train_start
        conditional_log(self.logger, self.process_rank, 'Training time: {:.4f}s'.format(train_time))
        if self.total_iters > 0:
            self.time_per_batch = float(total_time_per_batch)/self.total_iters
        else:
            self.time_per_batch = 0.0
        conditional_log(self.logger, self.process_rank, 'Time per batch: {:.4f}s'.format(self.time_per_batch))
        if self.early_stop and self.dev_score_history:
            if len(self.dev_score_history) > 0:
                self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history)
            else:
                self.time_per_dev_pass = 0
            conditional_log(self.logger, self.process_rank, 'Time per dev pass: {:4f}s'.format(self.time_per_dev_pass))
        
        # Save the learnt model: save both the final model and the best model.
        # https://stackoverflow.com/a/43819235/3262406
        if self.process_rank == 0:
            self.save_function(model=self.model, save_path=self.model_path, model_suffix='final')
            conditional_log(self.logger, self.process_rank, 'Best model; Epoch {:d}; Iteration {:d}; Dev loss: {:.4f}'
                            .format(best_epoch, best_iter, best_dev_score))
            # self.model.load_state_dict(best_params)
            # self.save_function(model=self.model, save_path=self.model_path, model_suffix='best')
    
    @staticmethod
    def compute_loss(loss_components):
        """
        Models will return dict with different loss components, use this and compute batch loss.
        :param loss_components: dict('str': Variable)
        :return:
        """
        raise NotImplementedError


class BasicRankingTrainerDDP(GenericTrainerDDP):
    def __init__(self, cml_exp, logger, process_rank, num_gpus, model, batcher, model_path, data_path,
                 train_hparams, early_stop=True, verbose=True, dev_score='loss'):
        """
        Trainer for any model returning a ranking loss. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainerDDP.__init__(self, cml_exp=cml_exp, logger=logger, process_rank=process_rank, num_gpus=num_gpus,
                                   model=model, batcher=batcher, model_path=model_path,
                                   train_hparams=train_hparams, early_stop=early_stop, verbose=verbose,
                                   dev_score=dev_score)
        # Expect the presence of a directory with as many shuffled copies of the
        # dataset as there are epochs and a negative examples file.
        self.train_fnames = []
        # Expect these to be there for the case of using diff kinds of training data for the same
        # model; hard negatives models, different alignment models and so on.
        if 'train_suffix' in train_hparams:
            suffix = train_hparams['train_suffix']
            train_basename = 'train-{:s}'.format(suffix)
            dev_basename = 'dev-{:s}'.format(suffix)
        else:
            train_basename = 'train'
            dev_basename = 'dev'
        for i in range(self.num_epochs):
            # Each run contains a copy of shuffled data for itself
            # Each process gets a part of the data to consume in training.
            ex_fname = {
                'pos_ex_fname': os.path.join(model_path, 'shuffled_data', f'{train_basename}-{process_rank}-{i}.jsonl'),
            }
            self.train_fnames.append(ex_fname)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, '{:s}.jsonl'.format(dev_basename)),
        }
        # The split command in bash is asked to make exactly equal sized splits with
        # the remainder in a final file which is unused
        self.num_train = train_hparams['train_size']//num_gpus
        if self.num_train > self.batch_size:
            self.num_batches = int(np.ceil(float(self.num_train)/self.batch_size))
        else:
            self.num_batches = 1
        self.total_iters = self.num_epochs*self.num_batches
        if self.lr_decay_method == 'exponential':
            self.decay_lr_by = train_hparams['decay_lr_by']
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                              gamma=self.decay_lr_by)
        elif self.lr_decay_method == 'warmuplin':
            self.num_warmup_steps = train_hparams['num_warmup_steps']//num_gpus
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps,
                # Total number of training batches.
                num_training_steps=self.num_epochs*self.num_batches)
        elif self.lr_decay_method == 'warmupcosine':
            self.num_warmup_steps = train_hparams['num_warmup_steps']//num_gpus
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_epochs*self.num_batches)
        else:
            raise ValueError('Unknown lr_decay_method: {:}'.format(train_hparams['lr_decay_method']))
        # Every subclass needs to set this.
        self.loss_function_cal = BasicRankingTrainer.compute_loss
    
    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('rankl': rank loss value)
        :return: Variable.
        """
        return loss_components['rankl']
