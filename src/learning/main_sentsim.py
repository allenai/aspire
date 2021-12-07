"""
For the faceted similarity models:
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful, predict with trained model and run
evaluation
"""
import argparse, os, sys
import re
import logging
import codecs, pprint, json
import comet_ml as cml
from transformers import AutoConfig

import numpy as np
import torch
from torch.nn import functional
from . import data_utils, batchers, trainer
from .facetid_models import sentsim_models


def train_model(model_name, data_path, config_path, run_path, cl_args):
    """
    Read the int training and dev data, initialize and train the model.
    :param model_name: string; says which model to use.
    :param data_path: string; path to the directory with unshuffled data
        and the test and dev json files.
    :param config_path: string; path to the directory json config for model
        and trainer.
    :param run_path: string; path for shuffled training data for run and
        to which results and model gets saved.
    :param cl_args: argparse command line object.
    :return: None.
    """
    run_name = os.path.basename(run_path)
    # Load label maps and configs.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)
    
    cml_experiment = cml.Experiment(project_name='2021-ai2-scisim')
    cml_experiment.log_parameters(all_hparams)
    cml_experiment.set_name(run_name)
    cml_experiment.add_tags([cl_args.dataset, cl_args.model_name])
    
    # Unpack hyperparameter settings.
    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))
    
    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)
    
    # Initialize model.
    if model_name == 'cosentbert':
        model = sentsim_models.SentBERTWrapper(model_name=all_hparams['base-pt-layer'])
    elif model_name == 'ictsentbert':
        model = sentsim_models.ICTBERTWrapper(model_name=all_hparams['base-pt-layer'])
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    # Model class internal logic uses the names at times so set this here so it
    # is backward compatible.
    model.model_name = model_name
    logging.info(model)
    # Save an untrained model version.
    trainer.sentbert_save_function(model=model, save_path=run_path, model_suffix='init')
    
    # Move model to the GPU.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')
    
    # Initialize the trainer.
    if model_name in ['cosentbert', 'ictsentbert']:
        batcher_cls = batchers.SentTripleBatcher
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    
    if model_name in ['cosentbert']:
        model_trainer = trainer.BasicRankingTrainer(cml_exp=cml_experiment,
                                                    model=model, batcher=batcher_cls, data_path=data_path, model_path=run_path,
                                                    early_stop=True, dev_score='loss', train_hparams=all_hparams)
        model_trainer.save_function = trainer.sentbert_save_function
    elif model_name in ['ictsentbert']:
        model_trainer = trainer.BasicRankingTrainer(cml_exp=cml_experiment,
                                                    model=model, batcher=batcher_cls, data_path=data_path, model_path=run_path,
                                                    early_stop=True, dev_score='loss', train_hparams=all_hparams)
        model_trainer.save_function = trainer.ictbert_save_function
    # Train and save the best model to model_path.
    model_trainer.train()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser('train_model')
    # Where to get what.
    train_args.add_argument('--model_name', required=True,
                            choices=['cosentbert', 'ictsentbert'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['s2orccompsci', 's2orcbiomed'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--data_path', required=True,
                            help='Path to the jsonl dataset.')
    train_args.add_argument('--run_path', required=True,
                            help='Path to directory to save all run items to.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    train_args.add_argument('--log_fname',
                            help='File name for the log file to which logs get'
                                 ' written.')
    cl_args = parser.parse_args()
    # If a log file was passed then write to it.
    try:
        logging.basicConfig(level='INFO', format='%(message)s',
                            filename=cl_args.log_fname)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    # Else just write to stdout.
    except AttributeError:
        logging.basicConfig(level='INFO', format='%(message)s',
                            stream=sys.stdout)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    
    if cl_args.subcommand == 'train_model':
        train_model(model_name=cl_args.model_name, data_path=cl_args.data_path,
                    run_path=cl_args.run_path, config_path=cl_args.config_path, cl_args=cl_args)


if __name__ == '__main__':
    main()
