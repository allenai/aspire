"""
Process the RELISH dataset.
"""
import os
import codecs
import json
import collections
import csv
import sys

import pandas as pd
import spacy

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def scidocs2myjson(in_path, out_path, dataset_name):
    """
    - Write out jsonl file of abstracts and title.
    - Write out the annotations in a json.
    - Write out a csv file of the queries metadata.
    - Write out json file of the evaluation splits.
    :param in_path: directory with json file of title and abstracts for papers
        and subdir with annotations.
    :param in_path: directory with annotation and split text files.
    :param dataset_name: {'cite', 'cocite', 'coread', 'coview'}
    :return:
    """
    print(f'Dataset: {dataset_name}')
    # Read json file of paper data.
    with codecs.open(os.path.join(in_path, 'paper_metadata_view_cite_read.json'), 'r', 'utf-8') as fp:
        pid2paper_data = json.load(fp)
    # Read splits and relevance labels.
    qpids2pool = collections.defaultdict(list)
    dev_qpids, test_qpids = set(), set()
    allpid2data = {}
    invalid_queries = set()
    missing_cands = set()
    for split in ['val', 'test']:
        with codecs.open(os.path.join(in_path, dataset_name, f'{split}.qrel'), 'r', 'utf-8') as val_file:
            for line in val_file:
                items = line.strip().split()
                qpid, _, cand_pid, relevance = str(items[0]), items[1], str(items[2]), int(items[3])
                try:
                    assert(pid2paper_data[qpid]['abstract'] != None)
                    assert(pid2paper_data[qpid]['title'] != None)
                except (AssertionError, KeyError):
                    invalid_queries.add(qpid)
                    continue
                try:
                    assert(pid2paper_data[cand_pid]['abstract'] != None)
                    assert(pid2paper_data[cand_pid]['title'] != None)
                except (AssertionError, KeyError):
                    missing_cands.add(cand_pid)
                    continue
                allpid2data[cand_pid] = pid2paper_data[cand_pid]
                qpids2pool[qpid].append((cand_pid, relevance))
                allpid2data[qpid] = pid2paper_data[qpid]
                if split == 'val':
                    dev_qpids.add(qpid)
                else:
                    test_qpids.add(qpid)
    print(f'Invalid queries: {len(invalid_queries)}')
    print(f'Missing candidates: {len(missing_cands)}')
    assert(len(set.intersection(dev_qpids, test_qpids)) == 0)
    print(f'Dev queries: {len(dev_qpids)}')
    print(f'Test queries: {len(test_qpids)}')
    print(f'All papers: {len(allpid2data)}')

    # Write out split files:
    eval_splits = {'dev': list(dev_qpids), 'test': list(test_qpids)}
    with codecs.open(os.path.join(out_path, f'scid{dataset_name}-evaluation_splits.json'), 'w', 'utf-8') as fp:
        json.dump(eval_splits, fp)
        print('Wrote: {:s}'.format(fp.name))

    # Write abstracts in jsonl file.
    out_file = codecs.open(os.path.join(out_path, f'abstracts-scid{dataset_name}.jsonl'), 'w', 'utf-8')
    pid2abstract = {}
    papers_without_abstracts = 0
    for pid, pdata in allpid2data.items():
        metadata = {'year': pdata['year']}
        try:
            abs_sentences = scispacy_model(pdata['abstract'],
                                           disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                    'lemmatizer', 'parser', 'ner'])
        except TypeError:
            papers_without_abstracts += 1
            continue
        abs_sentences = [sent.text for sent in abs_sentences.sents]
        title = pdata['title']
        assert(title and len(abs_sentences) > 0)
        doc_dict = {'title': title, 'abstract': abs_sentences, 'paper_id': pid, 'metadata': metadata}
        pid2abstract[pid] = doc_dict
        out_file.write(json.dumps(doc_dict)+'\n')
    print(f'Invalid documents: {papers_without_abstracts}')
    print(f'Docs with data: {len(pid2abstract)}')
    print(f'Wrote: {out_file.name}')
    out_file.close()
    
    # Build qpids2anns and write and queries metadata.
    query_meta_file = codecs.open(os.path.join(out_path, f'scid{dataset_name}-queries-release.csv'), 'w', 'utf-8')
    query_meta_csv = csv.DictWriter(query_meta_file, extrasaction='ignore',
                                    fieldnames=['paper_id', 'title'])
    query_meta_csv.writeheader()
    num_cands_perq = []
    qpmid2cands = {}
    for qpid, rel_pool in qpids2pool.items():
        cands = [i[0] for i in rel_pool]
        relevances = [i[1] for i in rel_pool]
        if cands:
            qpmid2cands[qpid] = {'cands': cands, 'relevance_adju': relevances}
            query_meta_csv.writerow({'title': pid2abstract[qpid]['title'], 'paper_id': qpid})
        # Check that there arent papers with multiple ratings.
        assert(len(set(cands)) == len(cands))
        num_cands_perq.append(len(cands))
    cand_summ = pd.DataFrame(num_cands_perq).describe()
    print('Candidates per query: {:}'.format(cand_summ))
    with codecs.open(os.path.join(out_path, f'test-pid2anns-scid{dataset_name}.json'), 'w') as fp:
        json.dump(qpmid2cands, fp)
        print('Wrote: {:s}'.format(fp.name))
    print('Wrote: {:}\n'.format(query_meta_file.name))
    query_meta_file.close()


if __name__ == '__main__':
    scidocs2myjson(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/scidocs/data',
                   out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/my_scidocs',
                   dataset_name='cite')
    scidocs2myjson(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/scidocs/data',
                   out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/my_scidocs',
                   dataset_name='cocite')
    scidocs2myjson(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/scidocs/data',
                   out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/my_scidocs',
                   dataset_name='coread')
    scidocs2myjson(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/scidocs/data',
                   out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/my_scidocs',
                   dataset_name='coview')
    