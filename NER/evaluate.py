from NER.eval_datasets import CSFCube
from NER.eval_models import get_model
import collections
import codecs
import os
import json
import logging

def evaluate(params):
    facet = params['facet']
    model_name = params['model']
    root_path = params['root_path']
    entity_threshold = params.get('entity_threshold')
    label_blacklist = params.get('entity_label_blacklist')
    print(f"Evaluating {model_name} on facet {facet}")
    model = get_model(model_name)
    dataset = CSFCube(root_path, entity_threshold, entity_blacklist_label=label_blacklist)
    test_pool = dataset.get_test_pool(facet)
    results = collections.defaultdict(list)
    for query_pid, query_pool in test_pool.items():
        query_data = dataset.get(pid=query_pid)
        query_rep = model.encode(query_data, facet=facet)

        candidate_pids = query_pool['cands']
        candidate_similarities = dict()
        for candidate_pid in candidate_pids:
            candidate_data = dataset.get(candidate_pid)
            candidate_rep = model.encode(candidate_data)
            similarity = model.get_similarity(query_rep, candidate_rep)
            candidate_similarities[candidate_pid] = similarity

        sorted_candidates = sorted(candidate_similarities.items(), key=lambda i: i[1], reverse=True)
        results[query_pid] = [(cpid, -1*sim) for cpid, sim in sorted_candidates]
        print(query_pid, sorted_candidates[0])

    write_results(results, root_path, model_name, facet, label_blacklist)

def write_results(results, root_path, model_name, facet, label_blacklist=None):
    if label_blacklist is None:
        model_dir = os.path.join(root_path, model_name)
    else:
        model_dir = os.path.join(root_path, model_name, f'blacklist_{label_blacklist[0]}')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    output_path = os.path.join(model_dir, f'test-pid2pool-csfcube-{model_name}-{facet}-ranked.json')
    with codecs.open(output_path, 'w', 'utf-8') as fp:
        json.dump(results, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


if __name__ == '__main__':
    CSFCUBE_DATA_PATH = '/homes/roik/PycharmProjects/aspire/datasets_raw/s2orccompsci/csfcube/'
    ALL_MODEL_NAMES = ['aspire_base',
                       'aspire_sentence',
                       'aspire_contextual',
                       'specter_base',
                       'specter_sentence']
    ALL_FACETS = ['background', 'method', 'result']
    params = {
        'model': 'aspire_sentence',
        'facet': 'result',
        'root_path': CSFCUBE_DATA_PATH
    }
    evaluate(params)