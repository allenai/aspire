"""
Generate rankings over randidates for queries for different datasets and trained models
or baselines. There are three types of functions here: one assumes a set of embeddings
from a model stored to disk and ranks based on distance/similarity metrics of these
embeddings, another type of function uses a more complex late interaction method for
scoring query and candidate, and a final type of function consumes data, embeds and caches
the reps in memory and computes scores for ranking. The last kind of function is used
most often in practice. For each of these type of function there are also variants for
faceted datasets and whole abstract datasets.
"""
import copy
import os
import sys
import logging
import time
import codecs, json
import argparse
import collections

import joblib
import torch
from sklearn import neighbors
from sklearn import metrics as skmetrics
import numpy as np
from scipy import spatial
from transformers import AutoModel, AutoTokenizer

from . import data_utils as du
from ..learning.facetid_models import disent_models
from ..learning import batchers

# https://stackoverflow.com/a/46635273/3262406
np.set_printoptions(suppress=True)


class TrainedScoringModel:
    """
    Class to initialize trained model, load precomputed reps, and score query candidate pairs.
    """
    def __init__(self, model_name, trained_model_path, model_version='cur_best'):
        # Load label maps and configs.
        with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
        # Init model:
        if model_name in {'miswordpolyenc'}:
            model = disent_models.WordSentAlignPolyEnc(model_hparams=all_hparams)
        else:
            raise ValueError(f'Unknown model: {model_name}')
        model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
        model.load_state_dict(torch.load(model_fname))
        logging.info(f'Scoring model: {model_fname}')
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
            logging.info('Running on GPU.')
        model.eval()
        self.model_name = model_name
        self.model = model
    
    def predict(self, query, cands):
        """
        Use trained model to return scores between query and candidate.
        :param query: numpy.array; num_sents x encoding_dim.
        :param cands: list(numpy.array); pool_depth(num_sents x encoding_dim)
        :return:
        """
        batch_size = 128
        cand_batch = []
        cand_scores = []
        pair_sm = []
        for ci, cand_sents in enumerate(cands):
            cand_batch.append(cand_sents)
            if ci % 1000 == 0:
                logging.info(f'Examples: {ci}/{len(cands)}')
            if len(cand_batch) == batch_size:
                score_dict = self.model.score(query_reps=query, cand_reps=cand_batch)
                cand_scores.extend(score_dict['batch_scores'].tolist())
                pair_sm.extend(score_dict['pair_scores'])
                cand_batch = []
        # Handle final few candidates.
        if cand_batch:
            score_dict = self.model.score(query_reps=query, cand_reps=cand_batch)
            cand_scores.extend(score_dict['batch_scores'].tolist())
            pair_sm.extend(score_dict['pair_scores'])
        ret_dict = {'cand_scores': cand_scores, 'pair_scores': pair_sm}
        return ret_dict


class CachingTrainedScoringModel:
    """
    Class to initialize trained model, build reps, cache them, and score query candidate pairs.
    """
    def __init__(self, model_name, trained_model_path, model_version='cur_best'):
        # Load label maps and configs.
        with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
        # Init model:
        if model_name in {'miswordbienc'}:
            model = disent_models.WordSentAlignBiEnc(model_hparams=all_hparams)
            batcher = batchers.AbsSentTokBatcher
        elif model_name in {'sbalisentbienc'}:
            model = disent_models.WordSentAbsSupAlignBiEnc(model_hparams=all_hparams)
            batcher = batchers.AbsSentTokBatcher
        elif model_name in {'cospecter'}:
            model = disent_models.MySPECTER(model_hparams=all_hparams)
            batcher = batchers.AbsTripleBatcher
        else:
            raise ValueError(f'Unknown model: {model_name}')
        model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
        model.load_state_dict(torch.load(model_fname))
        logging.info(f'Scoring model: {model_fname}')
        self.tokenizer = AutoTokenizer.from_pretrained(all_hparams['base-pt-layer'])
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
            logging.info('Running on GPU.')
        model.eval()
        self.model_name = model_name
        self.model = model
        self.batcher = batcher
        self.pid2model_reps = {}
    
    def save_cache(self, out_fname):
        """
        Saves the cache to disk in case we want to use it ever.
        """
        joblib.dump(self.pid2model_reps, out_fname, compress=('gzip', 3))
    
    def predict(self, query_pid, cand_pids, pid2abstract, facet='all'):
        """
        Use trained model to return scores between query and candidate.
        :param query_pid: string
        :param cand_pids: list(string)
        :param pid2abstract: dict(string: dict)
        :param facet: string; {'all', 'background', 'method', 'result'}
        :return:
        """
        # Gets reps of uncached documents.
        encode_batch_size = 32
        uncached_pids = [cpid for cpid in cand_pids if cpid not in self.pid2model_reps]
        if query_pid not in self.pid2model_reps: uncached_pids.append(query_pid)
        if uncached_pids:
            batch_docs = []
            batch_pids = []
            for i, pid in enumerate(uncached_pids):
                batch_docs.append({'TITLE': pid2abstract[pid]['title'],
                                   'ABSTRACT': pid2abstract[pid]['abstract']})
                batch_pids.append(pid)
                if i % 1000 == 0:
                    logging.info(f'Encoding: {i}/{len(uncached_pids)}')
                if len(batch_docs) == encode_batch_size:
                    batch_dict = self.batcher.make_batch(raw_feed={'query_texts': batch_docs},
                                                         pt_lm_tokenizer=self.tokenizer)
                    with torch.no_grad():
                        batch_rep_dicts = self.model.caching_encode(batch_dict)
                    assert(len(batch_pids) == len(batch_rep_dicts))
                    for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                        self.pid2model_reps[upid] = batch_reps
                    batch_docs = []
                    batch_pids = []
            if batch_docs:  # Last batch.
                batch_dict = self.batcher.make_batch(raw_feed={'query_texts': batch_docs},
                                                     pt_lm_tokenizer=self.tokenizer)
                with torch.no_grad():
                    batch_rep_dicts = self.model.caching_encode(batch_dict)
                assert(len(batch_pids) == len(batch_rep_dicts))
                for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                    self.pid2model_reps[upid] = batch_reps
        # Score documents based on reps.
        # Get query facet sent idxs.
        if facet != 'all':
            query_abs_labs = ['background_label' if lab == 'objective_label' else lab for lab
                              in pid2abstract[query_pid]['pred_labels']]
            qf_idxs = [i for i, l in enumerate(query_abs_labs) if f'{facet}_label' == l]
            query_rep = copy.deepcopy(self.pid2model_reps[query_pid])
            # Select only the query sentence reps.
            query_rep['sent_reps'] = query_rep['sent_reps'][qf_idxs, :]
        else:
            query_rep = self.pid2model_reps[query_pid]
        score_batch_size = 64
        cand_batch = []
        cand_scores = []
        pair_sm = []
        for ci, cpid in enumerate(cand_pids):
            cand_batch.append(self.pid2model_reps[cpid])
            if ci % 1000 == 0:
                logging.info(f'Scoring: {ci}/{len(cand_pids)}')
            if len(cand_batch) == score_batch_size:
                with torch.no_grad():
                    score_dict = self.model.caching_score(query_encode_ret_dict=query_rep,
                                                          cand_encode_ret_dicts=cand_batch)
                cand_scores.extend(score_dict['batch_scores'].tolist())
                pair_sm.extend(score_dict['pair_scores'])
                cand_batch = []
        if cand_batch:  # Handle final few candidates.
            with torch.no_grad():
                score_dict = self.model.caching_score(query_encode_ret_dict=query_rep,
                                                      cand_encode_ret_dicts=cand_batch)
            cand_scores.extend(score_dict['batch_scores'].tolist())
            pair_sm.extend(score_dict['pair_scores'])
        ret_dict = {'cand_scores': cand_scores, 'pair_scores': pair_sm}
        return ret_dict


def caching_scoringmodel_rank_pool_sentfaceted(root_path, trained_model_path, sent_rep_type,
                                               dataset, facet, run_name):
    """
    Given a pool of candidates re-rank the pool based on the model scores.
    Function for use when model classes provide methods to encode data, and then score
    documents. Representations are generated at the same time as scoringg, not apriori saved on disk.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string
    :param facet: string; {'background', 'method', 'result'} background and objective merged.
    :return: write to disk.
    """
    reps_path = os.path.join(root_path, sent_rep_type, run_name)
    # read candidate reps from the whole abstract reps and query reps from the faceted ones.
    pool_fname = os.path.join(root_path, f'test-pid2anns-{dataset}-{facet}.json')
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info(f'Read anns: {dataset}; total: {len(qpid2pool)}')
    # Load trained model.
    model = CachingTrainedScoringModel(model_name=sent_rep_type, trained_model_path=trained_model_path)
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, 'abstracts-csfcube-preds.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, f'{dataset}-{sent_rep_type}-ranked')
    du.create_dir(readable_dir_path)
    start = time.time()
    for qi, qpid in enumerate(query_pids):
        logging.info('Ranking query {:d}: {:s}'.format(qi, qpid))
        resfile = codecs.open(os.path.join(readable_dir_path, f'{qpid}-{dataset}-{sent_rep_type}-{facet}-ranked.txt'),
                              'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        ret_dict = model.predict(query_pid=qpid, cand_pids=cand_pids, pid2abstract=pid2abstract,
                                 facet='all' if sent_rep_type in {'cospecter'} else facet)
        cand_scores = ret_dict['cand_scores']
        pair_softmax = ret_dict['pair_scores']
        assert(len(cand_pids) == len(cand_scores))
        # Get nearest neighbours.
        cand2sims = {}
        cand_pair_sims_string = {}
        for cpid, cand_sim, pair_sent_sm in zip(cand_pids, cand_scores, pair_softmax):
            cand2sims[cpid] = cand_sim
            if isinstance(pair_sent_sm, list):
                mat = '\n'.join([np.array2string(np.around(t, 4), precision=3) for t in pair_sent_sm])
            else:
                mat = np.array2string(pair_sent_sm, precision=3)
            cand_pair_sims_string[cpid] = '{:.4f}\n{:s}'.format(cand_sim, mat)
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand2sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            ranked_pair_sim_strings.append(cand_pair_sims_string[cpid])
            query2rankedcands[qpid].append((cpid, sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    logging.info('Ranking candidates took: {:.4f}s'.format(time.time()-start))
    model.save_cache(out_fname=os.path.join(reps_path, f'pid2model_reps-{dataset}-{sent_rep_type}-{facet}.pickle'))
    with codecs.open(os.path.join(reps_path, f'test-pid2pool-{dataset}-{sent_rep_type}-{facet}-ranked.json'),
                     'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def caching_scoringmodel_rank_pool_sent(root_path, trained_model_path, sent_rep_type,
                                        dataset, run_name):
    """
    Given a pool of candidates re-rank the pool based on the model scores.
    Function for use when model classes provide methods to encode data, and then score
    documents. Representations are generated at the same time as scoringg, not apriori saved on disk.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string; {'sbtinybertsota', 'sbrobertanli'}
    :return: write to disk.
    """
    reps_path = os.path.join(root_path, sent_rep_type, run_name)
    pool_fname = os.path.join(root_path, f'test-pid2anns-{dataset}.json')
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info('Read anns: {:}; total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Load trained model.
    model = CachingTrainedScoringModel(model_name=sent_rep_type, trained_model_path=trained_model_path)
        
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, f'abstracts-{dataset}.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, f'{dataset}-{sent_rep_type}-ranked')
    du.create_dir(readable_dir_path)
    start = time.time()
    for qi, qpid in enumerate(query_pids):
        logging.info('Ranking query {:d}: {:s}'.format(qi, qpid))
        resfile = codecs.open(os.path.join(readable_dir_path, f'{qpid}-{dataset}-{sent_rep_type}-ranked.txt'),
                              'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        ret_dict = model.predict(query_pid=qpid, cand_pids=cand_pids, pid2abstract=pid2abstract)
        cand_scores = ret_dict['cand_scores']
        pair_softmax = ret_dict['pair_scores']
        assert(len(cand_pids) == len(cand_scores))
        # Get nearest neighbours.
        cand2sims = {}
        cand_pair_sims_string = {}
        for cpid, cand_sim, pair_sent_sm in zip(cand_pids, cand_scores, pair_softmax):
            cand2sims[cpid] = cand_sim
            cand_pair_sims_string[cpid] = (cand_sim, pair_sent_sm)
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand2sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            if len(ranked_pair_sim_strings) < 110:
                pair_sent_sm = cand_pair_sims_string[cpid][1]
                if isinstance(pair_sent_sm, list):
                    mat = '\n'.join([np.array2string(np.around(t, 4), precision=3) for t in pair_sent_sm])
                else:
                    mat = np.array2string(pair_sent_sm, precision=3)
                string = '{:.4f}\n{:s}'.format(cand_pair_sims_string[cpid][0], mat)
                ranked_pair_sim_strings.append(string)
            query2rankedcands[qpid].append((cpid, sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    logging.info('Ranking candidates took: {:.4f}s'.format(time.time()-start))
    # model.save_cache(out_fname=os.path.join(reps_path, f'pid2model_reps-{dataset}-{sent_rep_type}.pickle'))
    with codecs.open(os.path.join(reps_path, f'test-pid2pool-{dataset}-{sent_rep_type}-ranked.json'),
                     'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def scoringmodel_rank_pool_sentfaceted(root_path, trained_model_path, sent_rep_type,
                                       data_to_read, dataset, facet, run_name):
    """
    Given vectors on disk and a pool of candidates re-rank the pool based on the sentence rep
    and the facet passed. Function for use when the pool candidate reps are part of the gorc
    datasets reps. All reps are sentence level - this function is mainly for use with sentence bert
    outputs.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string; {'sbtinybertsota', 'sbrobertanli'}
    :param data_to_read: string; {'sent'}
    :param facet: string; {'background', 'method', 'result'} background and objective merged.
    :return: write to disk.
    """
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
    # read candidate reps from the whole abstract reps and query reps from the faceted ones.
    pool_fname = os.path.join(root_path, f'test-pid2anns-{dataset}-{facet}.json')
    all_map_fname = os.path.join(reps_path, f'pid2idx-{dataset}-sent.json')
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_docsents2idx = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info(f'Read anns: {dataset}; total: {len(qpid2pool)}')
    # Read vector reps.
    all_doc_reps = np.load(os.path.join(reps_path, f'{dataset}-{data_to_read}.npy'))
    np.nan_to_num(all_doc_reps, copy=False)
    logging.info(f'Read {dataset} sent reps: {all_doc_reps.shape}')
    # Load trained model.
    if sent_rep_type in {'miswordpolyenc'}:
        model = TrainedScoringModel(model_name=sent_rep_type, trained_model_path=trained_model_path)
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, 'abstracts-csfcube-preds.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, f'{dataset}-{sent_rep_type}-ranked')
    du.create_dir(readable_dir_path)
    for qpid in query_pids:
        resfile = codecs.open(os.path.join(readable_dir_path, f'{qpid}-{dataset}-{sent_rep_type}-{facet}-ranked.txt'),
                              'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        # Get the query abstracts query facet sentence representations
        query_abs_labs = ['background_label' if lab == 'objective_label' else lab for lab
                          in pid2abstract[qpid]['pred_labels']]
        # query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(query_abs_labs) if f'{facet}_label' == l]
        query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(query_abs_labs)]
        query_idx = [all_docsents2idx[i] for i in query_sent_repids]
        query_fsent_rep = all_doc_reps[query_idx]
        if query_fsent_rep.shape[0] == 768:
            query_fsent_rep = query_fsent_rep.reshape(1, query_fsent_rep.shape[0])
        # Get representations of all sentences in the pool.
        candpool_sent_reps = []
        cand_lens = []
        for cpid in cand_pids:
            cand_abs_labs = ['background_label' if lab == 'objective_label' else lab for lab
                             in pid2abstract[cpid]['pred_labels']]
            cand_ids = [f'{cpid}-{i}' for i in range(len(cand_abs_labs))]
            cand_lens.append(len(cand_ids))
            cand_doc_idxs = [all_docsents2idx[csent_id] for csent_id in cand_ids]
            candpool_sent_reps.append(all_doc_reps[cand_doc_idxs, :])
        ret_dict = model.predict(query=query_fsent_rep, cands=candpool_sent_reps)
        cand_scores = ret_dict['cand_scores']
        pair_softmax = ret_dict['pair_scores']
        assert(len(cand_pids) == len(cand_scores))
        # Get nearest neighbours.
        cand2sims = {}
        cand_pair_sims_string = {}
        for cpid, cand_sim, pair_sent_sm in zip(cand_pids, cand_scores, pair_softmax):
            cand2sims[cpid] = cand_sim
            cand_pair_sims_string[cpid] = '{:.4f}\n{:s}'.format(cand_sim, np.array2string(pair_sent_sm, precision=2))
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand2sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            ranked_pair_sim_strings.append(cand_pair_sims_string[cpid])
            # Save a distance because its what prior things saved.
            query2rankedcands[qpid].append((cpid, -1*sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    with codecs.open(os.path.join(reps_path, f'test-pid2pool-{dataset}-{sent_rep_type}-{facet}-ranked.json'),
                     'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def scoringmodel_rank_pool_sent(root_path, trained_model_path, sent_rep_type,
                                data_to_read, dataset, run_name):
    """
    Given vectors on disk and a pool of candidates re-rank the pool based on the sentence rep
    and the facet passed. Function for use when the pool candidate reps are part of the gorc
    datasets reps. All reps are sentence level - this function is mainly for use with sentence bert
    outputs.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string; {'sbtinybertsota', 'sbrobertanli'}
    :param data_to_read: string; {'sent'}
    :param facet: string; {'background', 'method', 'result'} background and objective merged.
    :return: write to disk.
    """
    dataset, split = dataset, ''
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
    # read candidate reps from the whole abstract reps and query reps from the faceted ones.
    pool_fname = os.path.join(root_path, 'test-pid2anns-{:s}{:s}.json'.format(dataset, split))
    # Also allow experimentation with unfaceted reps.
    all_map_fname = os.path.join(reps_path, 'pid2idx-{:s}-sent.json'.format(dataset))
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_docsents2idx = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info('Read anns: {:}; total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Read vector reps.
    all_doc_reps = np.load(os.path.join(reps_path, '{:s}-{:s}.npy'.
                                        format(dataset, data_to_read)))
    np.nan_to_num(all_doc_reps, copy=False)
    logging.info('Read {:s} sent reps: {:}'.format(dataset, all_doc_reps.shape))
    # Load trained model.
    if sent_rep_type in {'miswordpolyenc'}:
        model = TrainedScoringModel(model_name=sent_rep_type, trained_model_path=trained_model_path)
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, f'abstracts-{dataset}.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, f'{dataset}-{sent_rep_type}-ranked')
    du.create_dir(readable_dir_path)
    for qi, qpid in enumerate(query_pids):
        logging.info('Ranking query {:d}: {:s}'.format(qi, qpid))
        resfile = codecs.open(os.path.join(readable_dir_path, f'{qpid}-{dataset}-{sent_rep_type}-ranked.txt'),
                              'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        # Get the query abstracts query facet sentence representations
        query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(pid2abstract[qpid]['abstract'])]
        query_idx = [all_docsents2idx[i] for i in query_sent_repids]
        query_fsent_rep = all_doc_reps[query_idx]
        if query_fsent_rep.shape[0] == 768:
            query_fsent_rep = query_fsent_rep.reshape(1, query_fsent_rep.shape[0])
        # Get representations of all sentences in the pool.
        candpool_sent_reps = []
        cand_lens = []
        for cpid in cand_pids:
            cand_ids = [f'{cpid}-{i}' for i in range(len(pid2abstract[cpid]['abstract']))]
            cand_lens.append(len(cand_ids))
            cand_doc_idxs = [all_docsents2idx[csent_id] for csent_id in cand_ids]
            candpool_sent_reps.append(all_doc_reps[cand_doc_idxs, :])
        ret_dict = model.predict(query=query_fsent_rep, cands=candpool_sent_reps)
        cand_scores = ret_dict['cand_scores']
        pair_softmax = ret_dict['pair_scores']
        assert(len(cand_pids) == len(cand_scores))
        # Get nearest neighbours.
        cand2sims = {}
        cand_pair_sims_string = {}
        for cpid, cand_sim, pair_sent_sm in zip(cand_pids, cand_scores, pair_softmax):
            cand2sims[cpid] = cand_sim
            cand_pair_sims_string[cpid] = (cand_sim, pair_sent_sm)
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand2sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            if len(ranked_pair_sim_strings) < 110:
                string = '{:.4f}\n{:s}'.format(cand_pair_sims_string[cpid][0],
                                               np.array2string(cand_pair_sims_string[cpid][1], precision=2))
                ranked_pair_sim_strings.append(string)
            # Save a distance because its what prior things saved.
            query2rankedcands[qpid].append((cpid, -1*sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    with codecs.open(os.path.join(reps_path, f'test-pid2pool-{dataset}-{sent_rep_type}-ranked.json'),
                     'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))
        

def print_one_pool_nearest_neighbours(qdocid, all_neighbour_docids, pid2paperdata, resfile, pid_sources,
                                      ranked_pair_sim_strings=None):
    """
    Given the nearest neighbours indices write out the title and abstract and
    if the neighbour is cited in the query.
    :return:
    """
    # Print out the nearest neighbours to disk.
    qtitle = pid2paperdata[qdocid]['title']
    resfile.write('======================================================================\n')
    try:
        year = pid2paperdata[qdocid]['metadata']['year']
        # -6 is because the label is named {:s}_label.format(facet) by the predictor.
        qabs = '\n'.join(['{:s}: {:s}'.format(facet[:-6], sent) for sent, facet in
                          zip(pid2paperdata[qdocid]['abstract'], pid2paperdata[qdocid]['pred_labels'])])
    except KeyError:
        year = None
        qabs = '\n'.join(['{:d}: {:s}'.format(i, sent) for i, sent in
                          enumerate(pid2paperdata[qdocid]['abstract'])])
    resfile.write('PAPER_ID: {:s}; YEAR: {:}\n'.format(qdocid, year))
    resfile.write('TITLE: {:s}\n'.format(qtitle))
    resfile.write('ABSTRACT:\n{:s}\n'.format(qabs))
    # This happens in the case of treccovid.
    if 'topic_narratives' in pid2paperdata[qdocid]:
        resfile.write('TOPIC-ID: {:s}\n'.format(pid2paperdata[qdocid]['topic_ids']))
        narratives = [tn['narrative'] for tn in pid2paperdata[qdocid]['topic_narratives']]
        resfile.write('TOPIC Narrative:\n{:s}\n'.format('\n'.join(narratives)))
    resfile.write('===================================\n')
    written_candidates = 0
    for ranki, (ndocid, sources) in enumerate(zip(all_neighbour_docids, pid_sources)):
        # Do this only for treccovid.
        if written_candidates > 100 and 'topic_narratives' in pid2paperdata[qdocid]:
            break
        # These are the two noise documents which trip people up. >_<
        if ndocid in {'5111924', '41022419'}: continue
        try:
            ntitle = pid2paperdata[ndocid]['title']
        except KeyError:
            continue
        try:
            nabs = '\n'.join(['{:s}: {:s}'.format(facet[:-6], sent) for sent, facet in
                              zip(pid2paperdata[ndocid]['abstract'], pid2paperdata[ndocid]['pred_labels'])])
            year = pid2paperdata[ndocid]['metadata']['year']
        except KeyError:
            year = None
            nabs = '\n'.join(['{:d}: {:s}'.format(i, sent) for i, sent in
                              enumerate(pid2paperdata[ndocid]['abstract'])])
        resfile.write('RANK: {:d}\n'.format(ranki))
        resfile.write('PAPER_ID: {:s}; YEAR: {:}\n'.format(ndocid, year))
        # This is either a list of strings or a int value of relevance.
        if isinstance(sources, list):
            resfile.write('sources: {:}\n'.format(', '.join(sources)))
        elif isinstance(sources, int):
            resfile.write('RELS: {:}\n'.format(sources))
        if ranked_pair_sim_strings:
            resfile.write('Query sent sims:\n{:}\n'.format(ranked_pair_sim_strings[ranki]))
        resfile.write('TITLE: {:s}\n'.format(ntitle))
        resfile.write('ABSTRACT:\n{:s}\n\n'.format(nabs))
        written_candidates += 1
    resfile.write('======================================================================\n')
    resfile.write('\n')


def rank_pool(root_path, sent_rep_type, data_to_read, dataset, run_name):
    """
    Given vectors on disk and a pool of candidates combined with gold citations
    re-rank the pool based on the whole abstract rep alone.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param sent_rep_type: string;
    :param data_to_read: string; {'abstract', 'title'}
    :param dataset: string;
    :return: write to disk.
    """
    dataset, split = dataset, ''
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
    pool_fname = os.path.join(root_path, 'test-pid2anns-{:s}.json'.format(dataset))
    all_map_fname = os.path.join(reps_path, 'pid2idx-{:s}-{:s}.json'.format(dataset, data_to_read))
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    # Read doc2idx maps.
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_doc2idx = json.load(fp)
    logging.info('Read maps {:s}: {:}'.format(dataset, len(all_doc2idx)))
    # Get queries pids (all queries have a required document)
    logging.info('Read map all: {:}; total queries: {:}'.
                 format(len(all_doc2idx), len(qpid2pool)))
    # Read vector reps.
    if sent_rep_type in {'specter', 'cospecter'}:
        all_doc_reps = np.load(os.path.join(reps_path, '{:s}-{:s}s.npy'.format(dataset, data_to_read)))
        np.nan_to_num(all_doc_reps, copy=False)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info('Read anns: {:}; total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, f'abstracts-{dataset}.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, '{:s}{:s}-{:s}-ranked'.format(dataset, split, sent_rep_type))
    du.create_dir(readable_dir_path)
    for qi, qpid in enumerate(qpid2pool.keys()):
        logging.info('Ranking query {:d}: {:s}'.format(qi, qpid))
        resfile = codecs.open(os.path.join(readable_dir_path, '{:s}-{:s}{:s}-{:s}-ranked.txt'.
                                           format(qpid, dataset, split, sent_rep_type)), 'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        query_idx = all_doc2idx[qpid]
        query_rep = all_doc_reps[query_idx]
        if query_rep.shape[0] != 1:  # The sparse one is already reshaped somehow.
            query_rep = query_rep.reshape(1, query_rep.shape[0])
        pool_idxs = []
        for cpid in cand_pids:
            try:
                pool_idxs.append(all_doc2idx[cpid])
            except KeyError:
                continue
        pool_reps = all_doc_reps[pool_idxs, :]
        index = neighbors.NearestNeighbors(n_neighbors=len(pool_idxs), algorithm='brute')
        index.fit(pool_reps)
        # Get nearest neighbours.
        nearest_dists, nearest_idxs = index.kneighbors(X=query_rep)
        # Build the re-ranked list of paper_ids.
        neigh_ids = list(nearest_idxs[0])
        neigh_dists = list(nearest_dists[0])
        ranked_cand_pids = [cand_pids[nidx] for nidx in neigh_ids]
        ranked_cand_pid_rels = [cand_pid_rels[nidx] for nidx in neigh_ids]
        for nidx, ndist in zip(neigh_ids, neigh_dists):
            # cand_pids is a list of pids
            ndocid = cand_pids[nidx]
            if ndocid == qpid:
                # This should never happen but sometimes the gold cited data
                # contains the query id. hmmmmm.
                logging.info(qpid)
                continue
            query2rankedcands[qpid].append((ndocid, ndist))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels)
        resfile.close()
    with codecs.open(os.path.join(reps_path, 'test-pid2pool-{:s}-{:s}-ranked.json'.
            format(dataset, sent_rep_type)), 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def rank_pool_sent_treccovid(root_path, sent_rep_type, data_to_read, dataset, run_name):
    """
    Given vectors on disk and a pool of candidates re-rank the pool based on the sentence rep.
    Function for use when the pool candidate reps are part of the gorc
    datasets reps. All reps are sentence level - this function is mainly for use with sentence encoder
    outputs.
    This is a function to use when the candidate pools are deep (and have overlaps across queries)
    and indivudually going over every query is a waste.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string; {'sbtinybertsota', 'sbrobertanli'}
    :param data_to_read: string; {'sent'}
    :return: write to disk.
    """
    dataset, split = dataset, ''
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
        try:
            with codecs.open(os.path.join(reps_path, 'run_info.json'), 'r', 'utf-8') as fp:
                run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
            score_type = all_hparams['score_aggregation']
        except (FileNotFoundError, KeyError) as err:
            logging.info(f'Error loading run_info.json: {err}')
            score_type = 'cosine'
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
        score_type = 'cosine'
    logging.info(f'Score type: {score_type}')
    pool_fname = os.path.join(root_path, 'test-pid2anns-{:s}{:s}.json'.format(dataset, split))
    all_map_fname = os.path.join(reps_path, 'pid2idx-{:s}-sent.json'.format(dataset))
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_docsents2idx = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info('Read anns: {:}; total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Read vector reps.
    all_doc_reps = np.load(os.path.join(reps_path, '{:s}-{:s}.npy'.
                                        format(dataset, data_to_read)))
    np.nan_to_num(all_doc_reps, copy=False)
    logging.info('Read {:s} sent reps: {:}'.format(dataset, all_doc_reps.shape))
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, f'abstracts-{dataset}.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    logging.info(f'Abstracts: {len(pid2abstract)}')
    # Get ids for all the queries.
    all_query_sent_repids = []
    for qpid in query_pids:
        query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(pid2abstract[qpid]['abstract'])]
        all_query_sent_repids.extend(query_sent_repids)
    allqsentrep2idx = dict([(repid, idx) for idx, repid in enumerate(all_query_sent_repids)])
    all_query_idxs = [all_docsents2idx[i] for i in all_query_sent_repids]
    all_query_sent_reps = all_doc_reps[all_query_idxs, :]
    if score_type in {'dotlse'}:  # Dot product was used for training.
        allquery2cand_sims = np.matmul(all_query_sent_reps, all_doc_reps.T)
    elif score_type in {'l2lse', 'l2max', 'l2top2'}:
        allquery2cand_sims = -1.0*spatial.distance.cdist(all_query_sent_reps, all_doc_reps)
    elif score_type in {'cosinemax', 'cosine'}:
        allquery2cand_sims = skmetrics.pairwise.cosine_similarity(all_query_sent_reps, all_doc_reps)
    logging.info('All query cand sims: {:}'.format(allquery2cand_sims.shape))
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, '{:s}{:s}-{:s}-ranked'.format(dataset, split, sent_rep_type))
    du.create_dir(readable_dir_path)
    for qi, qpid in enumerate(query_pids):
        resfile = codecs.open(os.path.join(readable_dir_path, '{:s}-{:s}{:s}-{:s}-ranked.txt'.
                                           format(qpid, dataset, split, sent_rep_type)), 'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        # Get the query abstracts sentence representations
        query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(pid2abstract[qpid]['abstract'])]
        query_idx = [allqsentrep2idx[i] for i in query_sent_repids]
        # Get idxs of all sentences in the pool.
        pool_sent_ids = []
        cand_lens = []
        for cpid in cand_pids:
            cand_ids = [f'{cpid}-{i}' for i in range(len(pid2abstract[cpid]['abstract']))]
            cand_lens.append(len(cand_ids))
            pool_sent_ids.extend(cand_ids)
        pool_idxs = [all_docsents2idx[csent_id] for csent_id in pool_sent_ids]
        query2cand_sims = allquery2cand_sims[np.ix_(query_idx, pool_idxs)]
        logging.info('Ranking query {:d}: {:s}; {:}'.format(qi, qpid, query2cand_sims.shape))
        # Get nearest neighbours.
        start_idx = 0
        cand_sims = {}
        cand_pair_sims = {}
        for cpid, num_csents in zip(cand_pids, cand_lens):
            pair_sent_sims = query2cand_sims[:, start_idx: start_idx+num_csents]
            if score_type == 'l2top2':
                try:
                    # partial sort smallest distance to largest.
                    temp = np.partition(-1*pair_sent_sims.flatten(), kth=2)
                    # sum the top2 similarities.
                    max_sim = float(np.sum(-1*temp[:2]))
                # Some q-cand pairs have 2 or fewer sentences.
                except ValueError:
                    max_sim = float(np.sum(pair_sent_sims.flatten()))
            else:
                max_sim = float(np.max(pair_sent_sims))
            cand_sims[cpid] = max_sim
            cand_pair_sims[cpid] = pair_sent_sims
            start_idx += num_csents
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand_sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            # Only save these for the cands which you will print out.
            if len(ranked_pair_sim_strings) < 110:
                ranked_pair_sim_strings.append(np.array2string(cand_pair_sims[cpid], precision=2))
            # Save a distance because its what prior things saved.
            query2rankedcands[qpid].append((cpid, -1*sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    with codecs.open(os.path.join(reps_path, 'test-pid2pool-{:s}{:s}-{:s}-ranked.json'.
            format(dataset, split, sent_rep_type)), 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def rank_pool_sent(root_path, sent_rep_type, data_to_read, dataset, run_name):
    """
    Given vectors on disk and a pool of candidates re-rank the pool based on the sentence rep.
    Function for use when the pool candidate reps are part of the gorc
    datasets reps. All reps are sentence level - this function is mainly for use with sentence encoder
    outputs.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string; {'sbtinybertsota', 'sbrobertanli'}
    :param data_to_read: string; {'sent'}
    :return: write to disk.
    """
    dataset, split = dataset, ''
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
        try:
            with codecs.open(os.path.join(reps_path, 'run_info.json'), 'r', 'utf-8') as fp:
                run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
            score_type = all_hparams['score_aggregation']
        except (FileNotFoundError, KeyError) as err:
            logging.info(f'Error loading run_info.json: {err}')
            score_type = 'cosine'
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
        score_type = 'cosine'
    logging.info(f'Score type: {score_type}')
    # read candidate reps from the whole abstract reps and query reps from the faceted ones.
    pool_fname = os.path.join(root_path, 'test-pid2anns-{:s}{:s}.json'.format(dataset, split))
    # Also allow experimentation with unfaceted reps.
    all_map_fname = os.path.join(reps_path, 'pid2idx-{:s}-sent.json'.format(dataset))
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_docsents2idx = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info('Read anns: {:}; total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Read vector reps.
    all_doc_reps = np.load(os.path.join(reps_path, '{:s}-{:s}.npy'.
                                        format(dataset, data_to_read)))
    np.nan_to_num(all_doc_reps, copy=False)
    logging.info('Read {:s} sent reps: {:}'.format(dataset, all_doc_reps.shape))
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, f'abstracts-{dataset}.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, '{:s}{:s}-{:s}-ranked'.format(dataset, split, sent_rep_type))
    du.create_dir(readable_dir_path)
    for qi, qpid in enumerate(query_pids):
        logging.info('Ranking query {:d}: {:s}'.format(qi, qpid))
        resfile = codecs.open(os.path.join(readable_dir_path, '{:s}-{:s}{:s}-{:s}-ranked.txt'.
                                           format(qpid, dataset, split, sent_rep_type)), 'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        # Get the query abstracts sentence representations
        query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(pid2abstract[qpid]['abstract'])]
        query_idx = [all_docsents2idx[i] for i in query_sent_repids]
        query_fsent_rep = all_doc_reps[query_idx]
        if query_fsent_rep.shape[0] == 768:
            query_fsent_rep = query_fsent_rep.reshape(1, query_fsent_rep.shape[0])
        # Get representations of all sentences in the pool.
        pool_sent_ids = []
        cand_lens = []
        for cpid in cand_pids:
            cand_ids = [f'{cpid}-{i}' for i in range(len(pid2abstract[cpid]['abstract']))]
            cand_lens.append(len(cand_ids))
            pool_sent_ids.extend(cand_ids)
        pool_idxs = [all_docsents2idx[csent_id] for csent_id in pool_sent_ids]
        candpool_sent_reps = all_doc_reps[pool_idxs, :]
        if score_type in {'dotlse'}:  # Dot product was used for training.
            query2cand_sims = np.matmul(query_fsent_rep, candpool_sent_reps.T)
        elif score_type in {'l2lse', 'l2max', 'l2top2'}:
            query2cand_sims = -1.0*spatial.distance.cdist(query_fsent_rep, candpool_sent_reps)
        elif score_type in {'cosinemax', 'cosine'}:
            query2cand_sims = skmetrics.pairwise.cosine_similarity(query_fsent_rep, candpool_sent_reps)
        # Get nearest neighbours.
        start_idx = 0
        cand_sims = {}
        cand_pair_sims_string = {}
        for cpid, num_csents in zip(cand_pids, cand_lens):
            pair_sent_sims = query2cand_sims[:, start_idx: start_idx+num_csents]
            if score_type == 'l2top2':
                try:
                    # partial sort largest sim to smallest.
                    temp = np.partition(-1*pair_sent_sims.flatten(), kth=2)
                    # sum the top2 similarities.
                    max_sim = float(np.sum(-1*temp[:2]))
                # Some q-cand pairs have 2 or fewer sentences.
                except ValueError:
                    max_sim = float(np.sum(pair_sent_sims.flatten()))
            else:
                max_sim = float(np.max(pair_sent_sims))
            cand_sims[cpid] = max_sim
            cand_pair_sims_string[cpid] = np.array2string(pair_sent_sims, precision=2)
            start_idx += num_csents
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand_sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            ranked_pair_sim_strings.append(cand_pair_sims_string[cpid])
            # Save a distance because its what prior things saved.
            query2rankedcands[qpid].append((cpid, -1*sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    with codecs.open(os.path.join(reps_path, 'test-pid2pool-{:s}{:s}-{:s}-ranked.json'.
            format(dataset, split, sent_rep_type)), 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def rank_pool_sentfaceted(root_path, sent_rep_type, data_to_read, dataset, facet, run_name):
    """
    Given vectors on disk and a pool of candidates re-rank the pool based on the sentence rep
    and the facet passed. Function for use when the pool candidate reps are part of the gorc
    datasets reps. All reps are sentence level - this function is mainly for use with sentence bert
    outputs.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'csfcube'}; eval dataset to use.
    :param sent_rep_type: string; {'sbtinybertsota', 'sbrobertanli'}
    :param data_to_read: string; {'sent'}
    :param facet: string; {'background', 'method', 'result'} background and objective merged.
    :return: write to disk.
    """
    dataset, split = dataset, ''
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
        try:
            with codecs.open(os.path.join(reps_path, 'run_info.json'), 'r', 'utf-8') as fp:
                run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
            score_type = all_hparams['score_aggregation']
        except (FileNotFoundError, KeyError) as err:
            logging.info(f'Error loading run_info.json: {err}')
            score_type = 'cosine'
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
        score_type = 'cosine'
    logging.info(f'Score type: {score_type}')
    # read candidate reps from the whole abstract reps and query reps from the faceted ones.
    pool_fname = os.path.join(root_path, 'test-pid2anns-{:s}{:s}-{:s}.json'.format(dataset, split, facet))
    # Also allow experimentation with unfaceted reps.
    all_map_fname = os.path.join(reps_path, 'pid2idx-{:s}-sent.json'.format(dataset))
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_docsents2idx = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in qpid2pool]
    logging.info('Read anns: {:}; total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Read vector reps.
    all_doc_reps = np.load(os.path.join(reps_path, '{:s}-{:s}.npy'.format(dataset, data_to_read)))
    np.nan_to_num(all_doc_reps, copy=False)
    logging.info('Read {:s} sent reps: {:}'.format(dataset, all_doc_reps.shape))
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, 'abstracts-csfcube-preds.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, '{:s}{:s}-{:s}-ranked'.format(dataset, split, sent_rep_type))
    du.create_dir(readable_dir_path)
    for qpid in query_pids:
        resfile = codecs.open(os.path.join(readable_dir_path, '{:s}-{:s}{:s}-{:s}-{:s}-ranked.txt'.
                                           format(qpid, dataset, split, sent_rep_type, facet)), 'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        # Get the query abstracts query facet sentence representations
        query_abs_labs = ['background_label' if lab == 'objective_label' else lab for lab
                          in pid2abstract[qpid]['pred_labels']]
        query_sent_repids = [f'{qpid}-{i}' for i, l in enumerate(query_abs_labs) if f'{facet}_label' == l]
        query_idx = [all_docsents2idx[i] for i in query_sent_repids]
        query_fsent_rep = all_doc_reps[query_idx]
        if query_fsent_rep.shape[0] == 768:
            query_fsent_rep = query_fsent_rep.reshape(1, query_fsent_rep.shape[0])
        # Get representations of all sentences in the pool.
        pool_sent_ids = []
        cand_lens = []
        for cpid in cand_pids:
            cand_abs_labs = ['background_label' if lab == 'objective_label' else lab for lab
                             in pid2abstract[cpid]['pred_labels']]
            cand_ids = [f'{cpid}-{i}' for i in range(len(cand_abs_labs))]
            cand_lens.append(len(cand_ids))
            pool_sent_ids.extend(cand_ids)
        pool_idxs = [all_docsents2idx[csent_id] for csent_id in pool_sent_ids]
        candpool_sent_reps = all_doc_reps[pool_idxs, :]
        if score_type in {'dotlse'}:  # Dot product was used for training.
            query2cand_sims = np.matmul(query_fsent_rep, candpool_sent_reps.T)
        elif score_type in {'l2lse', 'l2max', 'l2top2'}:
            query2cand_sims = -1.0*spatial.distance.cdist(query_fsent_rep, candpool_sent_reps)
        elif score_type in {'cosinemax', 'cosine'}:
            query2cand_sims = skmetrics.pairwise.cosine_similarity(query_fsent_rep, candpool_sent_reps)
        # Get nearest neighbours.
        start_idx = 0
        cand_sims = {}
        cand_pair_sims_string = {}
        for cpid, num_csents in zip(cand_pids, cand_lens):
            pair_sent_sims = query2cand_sims[:, start_idx: start_idx+num_csents]
            if score_type == 'l2top2':
                # partial sort largest sim to smallest.
                temp = np.partition(-1*pair_sent_sims.flatten(), kth=2)
                # sum the top2 similarities.
                max_sim = float(np.sum(-1*temp[:2]))
            else:
                max_sim = float(np.max(pair_sent_sims))
            # elif score_method == 'maxsum':
            #     max_sim = float(np.sum(np.max(pair_sent_sims, axis=1)))
            # elif score_method == 'top3':
            #     flat = pair_sent_sims.flatten()
            #     topidx = np.argpartition(flat, -3)[-3:]
            #     max_sim = float(np.sum(flat[topidx]))
            # else:
            #     raise AssertionError
            cand_sims[cpid] = max_sim
            cand_pair_sims_string[cpid] = np.array2string(pair_sent_sims, precision=2)
            start_idx += num_csents
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand_sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            ranked_pair_sim_strings.append(cand_pair_sims_string[cpid])
            # Save a distance because its what prior things saved.
            query2rankedcands[qpid].append((cpid, -1*sim))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile,
                                          pid_sources=ranked_cand_pid_rels,
                                          ranked_pair_sim_strings=ranked_pair_sim_strings)
        resfile.close()
    with codecs.open(os.path.join(reps_path, 'test-pid2pool-{:s}{:s}-{:s}-{:s}-ranked.json'.
            format(dataset, split, sent_rep_type, facet)), 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def rank_pool_faceted(root_path, sent_rep_type, data_to_read, dataset, facet, run_name):
    """
    Given vectors on disk and a pool of candidates re-rank the pool based on the sentence rep
    and the facet passed. Function for use when the pool candidate reps are part of the gorc
    datasets reps. Query reps per facet will be on disk.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string;
    :param sent_rep_type: string;
    :param data_to_read: string; {'abstract', 'title'}
    :param facet: string; {'background', 'method', 'result'} backgroud and objective merged.
    :return: write to disk.
    """
    if run_name:
        reps_path = os.path.join(root_path, sent_rep_type, run_name)
    else:
        reps_path = os.path.join(root_path, sent_rep_type)
    # read candidate reps from the whole abstract reps and query reps from the faceted ones.
    pool_fname = os.path.join(root_path, f'test-pid2anns-{dataset}-{facet}.json')
    # Also allow experimentation with unfaceted reps.
    if sent_rep_type in {'cospecter'}:
        all_map_fname = os.path.join(reps_path, f'pid2idx-{dataset}-{data_to_read}.json')
    # Read test pool.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
    # Read doc2idx maps.
    with codecs.open(all_map_fname, 'r', 'utf-8') as fp:
        all_doc2idx = json.load(fp)
    query_pids = [qpid for qpid in qpid2pool.keys() if qpid in all_doc2idx]
    logging.info('Read maps facet {:s}: total: {:}; valid: {:}'.
                 format(dataset, len(qpid2pool), len(query_pids)))
    # Read vector reps.
    if sent_rep_type in {'cospecter', 'specter'}:
        all_doc_reps = np.load(os.path.join(reps_path, f'{dataset}-{data_to_read}s.npy'))
        np.nan_to_num(all_doc_reps, copy=False)
    logging.info('Read faceted {:s}: {:}'.format(dataset, all_doc_reps.shape))
    # Read in abstracts for printing readable.
    pid2abstract = {}
    with codecs.open(os.path.join(root_path, 'abstracts-csfcube-preds.jsonl'), 'r', 'utf-8') as absfile:
        for line in absfile:
            injson = json.loads(line.strip())
            pid2abstract[injson['paper_id']] = injson
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, '{:s}-{:s}-ranked'.format(dataset, sent_rep_type))
    du.create_dir(readable_dir_path)
    for qpid in query_pids:
        resfile = codecs.open(os.path.join(readable_dir_path, '{:s}-{:s}-{:s}-{:s}-ranked.txt'.
                                           format(qpid, dataset, sent_rep_type, facet)), 'w', 'utf-8')
        cand_pids = qpid2pool[qpid]['cands']
        cand_pid_rels = qpid2pool[qpid]['relevance_adju']
        query_idx = all_doc2idx[qpid]
        query_rep = all_doc_reps[query_idx]
        if query_rep.shape[0] != 1:  # The sparse one is already reshaped somehow.
            query_rep = query_rep.reshape(1, query_rep.shape[0])
        pool_idxs = [all_doc2idx[pid] for pid in cand_pids]
        pool_reps = all_doc_reps[pool_idxs, :]
        index = neighbors.NearestNeighbors(n_neighbors=len(pool_idxs), algorithm='brute')
        index.fit(pool_reps)
        # Get nearest neighbours.
        nearest_dists, nearest_idxs = index.kneighbors(X=query_rep)
        # Build the re-ranked list of paper_ids.
        neigh_ids = list(nearest_idxs[0])
        neigh_dists = list(nearest_dists[0])
        ranked_cand_pids = [cand_pids[nidx] for nidx in neigh_ids]
        ranked_cand_pid_rels = [cand_pid_rels[nidx] for nidx in neigh_ids]
        for nidx, ndist in zip(neigh_ids, neigh_dists):
            # cand_pids is a list of pids
            ndocid = cand_pids[nidx]
            if ndocid == qpid:
                continue
            query2rankedcands[qpid].append((ndocid, ndist))
        # Print out the neighbours.
        print_one_pool_nearest_neighbours(qdocid=qpid, all_neighbour_docids=ranked_cand_pids,
                                          pid2paperdata=pid2abstract, resfile=resfile, pid_sources=ranked_cand_pid_rels)
        resfile.close()
    with codecs.open(os.path.join(reps_path, 'test-pid2pool-{:s}-{:s}-{:s}-ranked.json'.
            format(dataset, sent_rep_type, facet)), 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Rank the pool for every query.
    dataset_rank_pool = subparsers.add_parser('rank_pool')
    dataset_rank_pool.add_argument('--root_path', required=True,
                                   help='Path with abstracts, sentence reps and citation info.')
    dataset_rank_pool.add_argument('--run_name', default=None,
                                   help='Path with trained sentence reps if using.')
    dataset_rank_pool.add_argument('--rep_type', required=True,
                                   choices=['sbtinybertsota', 'sbrobertanli', 'sentpubmedbert', 'sbmpnet1B',
                                            'cosentbert', 'ictsentbert', 'cospecter',
                                            'miswordbienc', 'supsimcse', 'unsupsimcse',
                                            'miswordpolyenc', 'sbalisentbienc'],
                                   help='The kind of rep to use for nearest neighbours.')
    dataset_rank_pool.add_argument('--model_path',
                                   help='Path to directory with trained model to use for getting scoring function.')
    dataset_rank_pool.add_argument('--dataset', required=True,
                                   choices=['csfcube', 'relish', 'treccovid',
                                            'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'],
                                   help='The dataset to predict for.')
    dataset_rank_pool.add_argument('--facet',
                                   choices=['background', 'method', 'result'],
                                   help='Facet of abstract to read from.')
    dataset_rank_pool.add_argument('--log_fname',
                                   help='File name for the log file to which logs get written.')
    dataset_rank_pool.add_argument('--caching_scorer', action="store_true", default=False)
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
    
    if cl_args.subcommand == 'rank_pool':
        if cl_args.rep_type in {'sbtinybertsota', 'sbrobertanli', 'cosentbert', 'ictsentbert',
                                'miswordbienc', 'supsimcse', 'unsupsimcse',
                                'miswordpolyenc', 'sbalisentbienc', 'sbmpnet1B'}:
            data_to_read = 'sent'
        else:
            data_to_read = 'abstract'
        if cl_args.dataset in {'csfcube'}:
            if cl_args.rep_type in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B', 'cosentbert', 'ictsentbert',
                                    'miswordbienc', 'supsimcse', 'unsupsimcse', 'sbalisentbienc'} \
                    and not cl_args.caching_scorer:
                rank_pool_sentfaceted(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                                      data_to_read=data_to_read, dataset=cl_args.dataset, facet=cl_args.facet,
                                      run_name=cl_args.run_name)
            elif cl_args.rep_type in {'miswordpolyenc'}:
                scoringmodel_rank_pool_sentfaceted(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                                                   data_to_read=data_to_read, dataset=cl_args.dataset,
                                                   facet=cl_args.facet,
                                                   run_name=cl_args.run_name, trained_model_path=cl_args.model_path)
            elif cl_args.rep_type in {'cospecter', 'sbalisentbienc', 'miswordbienc'} \
                    and cl_args.caching_scorer:
                caching_scoringmodel_rank_pool_sentfaceted(
                    root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type, dataset=cl_args.dataset,
                    facet=cl_args.facet, run_name=cl_args.run_name, trained_model_path=cl_args.model_path)
            else:
                rank_pool_faceted(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                                  data_to_read=data_to_read, dataset=cl_args.dataset, facet=cl_args.facet,
                                  run_name=cl_args.run_name)
        elif cl_args.dataset in {'relish', 'treccovid', 'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'}:
            if cl_args.rep_type in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B', 'cosentbert',
                                    'ictsentbert', 'miswordbienc',
                                    'supsimcse', 'unsupsimcse', 'sbalisentbienc'} and \
                    not cl_args.caching_scorer and \
                    cl_args.dataset in {'relish', 'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'}:
                rank_pool_sent(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                               data_to_read=data_to_read, dataset=cl_args.dataset,
                               run_name=cl_args.run_name)
            elif cl_args.rep_type in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B', 'cosentbert',
                                      'ictsentbert', 'miswordbienc', 'supsimcse', 'unsupsimcse',
                                      'sbalisentbienc'} and \
                    not cl_args.caching_scorer and cl_args.dataset == 'treccovid':
                rank_pool_sent_treccovid(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                                         data_to_read=data_to_read, dataset=cl_args.dataset,
                                         run_name=cl_args.run_name)
            elif cl_args.rep_type in {'miswordpolyenc'}:
                scoringmodel_rank_pool_sent(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                                            data_to_read=data_to_read, dataset=cl_args.dataset,
                                            run_name=cl_args.run_name, trained_model_path=cl_args.model_path)
            elif cl_args.rep_type in {'cospecter', 'sbalisentbienc', 'miswordbienc'} \
                    and cl_args.caching_scorer:
                caching_scoringmodel_rank_pool_sent(
                    root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type, dataset=cl_args.dataset,
                    run_name=cl_args.run_name, trained_model_path=cl_args.model_path)
            else:
                rank_pool(root_path=cl_args.root_path, sent_rep_type=cl_args.rep_type,
                          data_to_read=data_to_read, dataset=cl_args.dataset, run_name=cl_args.run_name)


if __name__ == '__main__':
    main()
