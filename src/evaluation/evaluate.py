import argparse
import collections
from tqdm import tqdm
from src.evaluation.utils.models import get_model, SimilarityModel
from src.evaluation.utils.utils import *
from src.evaluation.utils.datasets import EvalDataset
from data_utils import create_dir
from typing import Union
import pandas as pd
from src.evaluation.utils.metrics import compute_metrics
import sys
import logging


def encode(model: SimilarityModel, dataset: EvalDataset):
    """
    Cache model encodings of an entire dataseth
    :param model: A model for encoding papers
    :param dataset: Dataset to encode
    """

    # get all dataset paper id which are uncached
    dataset_pids = dataset.dataset.keys()
    cached_pids = set() if model.cache is None else model.cache.keys()
    uncached_pids = set(dataset_pids).difference(set(cached_pids))
    uncached_ds = {k: dataset.get(k) for k in uncached_pids}

    # cache encodings
    if len(uncached_ds) > 0:
        logging.info(f"Encoding {len(uncached_ds)} uncached papers in {len(uncached_ds) // model.batch_size} batches")
        for batch_pids, batch_papers in tqdm((batchify(uncached_ds, model.batch_size))):
            model.cache_encodings(batch_pids, batch_papers)


def score(model: SimilarityModel,
          dataset: EvalDataset,
          facet: Union[str, None],
          scores_filename: str):
    """
    Calculate similarity scores between queries and their candidates in a test pool
    :param model: Model to test
    :param dataset: Dataset to take test pool from
    :param facet: Facet of query to use. Relevant only to CSFcube dataset.
    :param scores_filename: Saves results here
    :return:
    """

    # load test pool
    test_pool = dataset.get_test_pool(facet=facet)

    log_msg = f"Scoring {len(test_pool)} queries in {dataset.name}"
    if facet is not None:
        log_msg += f', facet: {facet}'
    logging.info(log_msg)

    # Get model similarity scores between each query and its candidates
    results = collections.defaultdict(list)
    for query_pid, query_pool in tqdm(list(test_pool.items())):

        # get query encoding
        # if faceted, also filter the encoding by facet
        query_encoding = model.get_encoding(pids=[query_pid], dataset=dataset)[query_pid]
        if facet is not None:
            query_encoding = model.get_faceted_encoding(query_encoding, facet, dataset.get(query_pid))

        # get candidates encoding
        candidate_pids = query_pool['cands']
        candidate_encodings = model.get_encoding(pids=candidate_pids, dataset=dataset)

        # For calculate similarities of each candidate to query encoding
        candidate_similarities = dict()
        for candidate_pid in candidate_pids:
            similarity = model.get_similarity(query_encoding, candidate_encodings[candidate_pid])
            candidate_similarities[candidate_pid] = similarity
        # sort candidates by similarity, ascending (lower score == closer encodings)
        sorted_candidates = sorted(candidate_similarities.items(), key=lambda i: i[1], reverse=True)
        results[query_pid] = [(cpid, -1*sim) for cpid, sim in sorted_candidates]

    # write scores
    with codecs.open(scores_filename, 'w', 'utf-8') as fp:
        json.dump(results, fp)
        logging.info(f'Wrote: {scores_filename}')


def evaluate(results_dir: str,
             facet: Union[str, None],
             dataset: EvalDataset,
             comet_exp_key=None):
    """
    Compute metrics based on a model's similarity scores on a dataset's test pool
    Assumes score() has already been called with relevant model_name, dataset and facet
    :param results_dir:
    :param model_name:
    :param facet:
    :param dataset:
    :param comet_exp_key:
    :return:
    """
    logging.info('Computing metrics')

    # load score results
    results = dict()
    if facet == 'all':
        for facet_i in FACETS:
            results[facet_i] = load_score_results(results_dir, dataset, facet_i)
    else:
        facet_key = 'unfaceted' if facet is None else facet
        results[facet_key] = load_score_results(results_dir, dataset, facet)

    # get queries metadata
    query_metadata = dataset.get_query_metadata()
    query_test_dev_split = dataset.get_test_dev_split()
    threshold_grade = dataset.get_threshold_grade()

    # compute metrics per query
    metrics = []
    metric_columns = None
    for facet_i, facet_results in results.items():
        for query_id, sorted_relevancies in facet_results.items():
            query_metrics = compute_metrics(sorted_relevancies,
                                            pr_atks=[5, 10, 20],
                                            threshold_grade=threshold_grade)
            if metric_columns is None:
                metric_columns = list(query_metrics.keys())
            query_metrics['facet'] = facet_i
            query_metrics['split'] = 'test' if query_test_dev_split is None else query_test_dev_split[query_id]
            query_metrics['paper_id'] = query_id
            query_metrics['title'] = query_metadata.loc[query_id]['title']
            metrics.append(query_metrics)
    metrics = pd.DataFrame(metrics)

    # write evaluations file per query
    query_metrics_filename = get_evaluations_filename(results_dir, facet, aggregated=False)
    metrics.to_csv(query_metrics_filename, index=False)
    logging.info(f'Wrote: {query_metrics_filename}')

    # aggergate metrics per (facet, dev/test_split)
    aggregated_metrics = []
    for facet_i in metrics.facet.unique():
        for split in metrics.split.unique():
            agg_results = metrics[(metrics.facet == facet_i) & (metrics.split == split)][metric_columns].mean().round(4).to_dict()
            logging.info(f'----------Results for {split}/{facet_i}----------')
            logging.info('\n'.join([f'{k}\t{agg_results[k]}' for k in ('av_precision', 'ndcg%20')]))
            agg_results['facet'] = facet_i
            agg_results['split'] = split
            aggregated_metrics.append(agg_results)
    if facet == 'all':
        for split in metrics.split.unique():
            agg_results = metrics[metrics.split == split][metric_columns].mean().round(4).to_dict()
            logging.info(f'----------Results for {split}/{facet}----------')
            logging.info('\n'.join([f'{k}\t{agg_results[k]}' for k in ('av_precision', 'ndcg%20')]))
            agg_results['facet'] = facet
            agg_results['split'] = split
            aggregated_metrics.append(agg_results)
    aggregated_metrics = pd.DataFrame(aggregated_metrics)

    # Write evaluation file aggregated per (facet, dev/test_split)
    aggregated_metrics_filename = get_evaluations_filename(results_dir, facet, aggregated=True)
    aggregated_metrics.to_csv(aggregated_metrics_filename, index=False)
    logging.info(f'Wrote: {aggregated_metrics_filename}')



def main(args):

    # init log
    if args.log_fname is not None:
        log_dir = os.path.split(os.path.join(os.getcwd(), args.log_fname))[0]
        if not os.path.exists(log_dir):
            create_dir(log_dir)
        logging.basicConfig(level='INFO', format='%(message)s', filename=args.log_fname)
    else:
        logging.basicConfig(level='INFO', format='%(message)s', stream=sys.stdout)

    # check validity of command-line arguments
    check_args(args)

    # init results dir
    results_dir = get_results_dir(args.results_dir, args.dataset_name, args.model_name, args.run_name)
    if not os.path.exists(results_dir):
        create_dir(results_dir)

    # init model and dataset
    dataset = EvalDataset(name=args.dataset_name, root_path=args.dataset_dir)
    model= None
    if 'encode' in args.actions or 'score' in args.actions:
        logging.info(f'Loading model: {args.model_name}')
        model = get_model(model_name=args.model_name)
        logging.info(f'Loading dataset: {args.dataset_name}')
        if args.cache:
            # init cache
            encodings_filename = get_encodings_filename(results_dir)
            logging.info(f'Setting model cache at: {encodings_filename}')
            model.set_encodings_cache(encodings_filename)

    if 'encode' in args.actions:
        # cache model's encodings of entire dataset
        encode(model, dataset)

    if 'score' in args.actions:
        # score model on dataset's test pool
        if args.facet == 'all':
            for facet in FACETS:
                score(model, dataset, facet=facet, scores_filename=get_scores_filename(results_dir, facet=facet))
        else:
            score(model, dataset, facet=args.facet, scores_filename=get_scores_filename(results_dir, facet=args.facet))

    if 'evaluate' in args.actions:
        # evaluate metrics for model scores
        evaluate(results_dir,
                 facet=args.facet,
                 dataset=dataset)

def check_args(args):
    if args.facet is not None:
        assert args.dataset_name == 'csfcube', f'Faceted query search is only tested on csfcube, not {args.dataset_name}'
    if args.dataset_name == 'csfcube' and args.facet is None:
        logging.info("No facet selected for CSFcube. Running on all facets.")
        args.facet = 'all'
    if 'encode' in args.actions and not args.cache:
        logging.info("Action 'encode' selected, automatically enabling cache")
        args.cache = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='The name of the model to run. Choose from a model_name '
                                                            'with an implementation in evaluation_models.get_model')
    parser.add_argument('--dataset_name', required=True, help='Dataset to evaluate similarities on',
                        choices=['gorcmatscicit', 'csfcube', 'relish', 'treccovid',
                                 'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'])
    parser.add_argument('--dataset_dir', required=True,
                        help="Dir to dataset files (e.g. abstracts-{dataset}.jsonl)")
    parser.add_argument('--results_dir', required=True,
                        help="Results base dir to store encodings cache, scores and metrics")
    parser.add_argument('--facet', help='Relevant only to csfcube dataset. Select a facet to use for the task'
                                        ' of faceted similarity search. If "all", tests all facets one at a time.',
                        choices=['result', 'method', 'background', 'all'], default=None)
    parser.add_argument('--cache', action='store_true', help='Use if we would like to cache the encodings of papers.'
                                                             'If action "encode" is selected, this is set automatically to true.')
    parser.add_argument('--run_name',help='Name of this evaluation run.\n'
                                          'Saves results under results_dir/model_name/run_name/\n'
                                          'to allow different results to same model_name')
    parser.add_argument('--trained_model_path', help='Basename for a trained model we would like to evaluate on.')
    parser.add_argument('--log_fname', help='Filename of log file')
    parser.add_argument('--actions', choices=['encode', 'score', 'evaluate'],
                        nargs="+", default=['encode', 'score', 'evaluate'],
                        help="""'Encode' creates vector representations for the entire dataset.
                        'Score' calculates similarity scores on the dataset's test pool.
                        'Evaluate' calculates metrics based on the similarity scores predicted.
                        By default does all three.""")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)