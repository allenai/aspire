import codecs
import os
import json
from src.evaluation.utils.datasets import EvalDataset
from data_utils import create_dir
from typing import Dict

FACETS = ('background', 'method', 'result')

def batchify(dataset: Dict, batch_size: int):
    """
    Splits dataset into batch size groups
    :param dataset: dict of {pid: paper_info}
    :param batch_size: batch size
    :return: Iterator which returns examples in batches of batch_size
    """
    pids = []
    batch = []
    for pid, data in dataset.items():
        pids.append(pid)
        batch.append(data)
        if len(batch) == batch_size:
            yield pids, batch
            pids = []
            batch = []
    if len(batch) > 0:
        yield pids, batch

def get_scores_filepath(root_path, model_name, run_name, dataset, facet):
    if facet is None:
        filename = f'test-pid2pool-{dataset}-{model_name}-ranked.json'
    else:
        filename = f'test-pid2pool-{dataset}-{model_name}-{facet}-ranked.json'
    scores_dir = os.path.join(root_path, model_name)
    if run_name is not None:
        scores_dir = os.path.join(scores_dir, run_name)
    return os.path.join(scores_dir, filename)

def get_cache_dir(cache_basedir,
                  dataset_name,
                  model_name,
                  run_name=None):
    # get path
    cache_path = os.path.join(cache_basedir, dataset_name, model_name)
    if run_name is not None:
        cache_path = os.path.join(cache_path, run_name)

    # create dir if path does not exist
    if not os.path.exists(cache_path):
        create_dir(cache_path)
    return cache_path

def get_results_dir(results_basedir, dataset_name, model_name, run_name):
    results_dir = os.path.join(results_basedir, dataset_name, model_name)
    if run_name is not None:
        results_dir = os.path.join(results_dir, run_name)
    return results_dir

def get_scores_filename(results_dir, facet):
    filename = 'scores.json' if facet is None else f'scores-{facet}.json'
    return os.path.join(results_dir, filename)

def get_encodings_filename(results_dir):
    return os.path.join(results_dir, 'encodings.h5')

def get_evaluations_filename(results_dir, facet, aggregated):
    metrics_type = 'aggregated' if aggregated else 'query'
    filename = f'{metrics_type}-evaluations.csv' if facet is None else f'{metrics_type}-evaluations-{facet}.csv'
    return os.path.join(results_dir, filename)

def load_score_results(results_dir, dataset: EvalDataset, facet):
    # get gold data relevances for facet
    gold_test_data = dataset.get_gold_test_data(facet)
    # get model similarity scores for facet
    with codecs.open(get_scores_filename(results_dir, facet), 'r', 'utf-8') as fp:
        model_scores = json.load(fp)
    results = {}
    for query_id, candidate_scores in model_scores.items():
        # get relevancy of each candidate, sorted by the model's similarity score to the query
        sorted_candidate_ids = [x[0] for x in candidate_scores]
        sorted_relevancies = [gold_test_data[query_id][pid] for pid in sorted_candidate_ids]
        results[query_id] = sorted_relevancies
    return results