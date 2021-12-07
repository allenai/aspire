"""
Process the RELISH dataset.
"""
import os
import codecs
import json
import csv
import pandas as pd
import random
import spacy

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def annotation_pmids(in_path):
    """
    Write out pmids of the RELISH documents.
    :param in_path:
    :return:
    """
    with codecs.open(os.path.join(in_path, 'RELISH_v1_ann.json'), 'r', 'utf-8') as fp:
        ann_dicts = json.load(fp)
        
    dataset_pmids = set()
    dataset_pmids_rep = []
    for ann_dict in ann_dicts:
        dataset_pmids.add(ann_dict['pmid'])
        dataset_pmids_rep.append(ann_dict['pmid'])
        dataset_pmids.update(ann_dict['response']['relevant'])
        dataset_pmids_rep.extend(ann_dict['response']['relevant'])
        dataset_pmids.update(ann_dict['response']['partial'])
        dataset_pmids_rep.extend(ann_dict['response']['partial'])
        dataset_pmids.update(ann_dict['response']['irrelevant'])
        dataset_pmids_rep.extend(ann_dict['response']['irrelevant'])
    print('All PMIDs: {:d}; Unique PMIDs: {:d}'.format(len(dataset_pmids_rep), len(dataset_pmids)))
    
    with codecs.open(os.path.join(in_path, 'RELISH_v1_pmids.txt'), 'w', 'utf-8') as fp:
        for pmid in dataset_pmids:
            fp.write('{:s}\n'.format(pmid))
        print('Wrote: {:s}'.format(fp.name))


def ann_stats2json(in_abs_path, in_path, out_path):
    """
    - Write out jsonl file of abstracts and title.
    :param in_abs_path: directory with title and abstracts for papers.
    :param in_path: directory with annotations json.
    :return:
    """
    filenames = os.listdir(in_abs_path)
    out_file = codecs.open(os.path.join(out_path, 'abstracts-relish.jsonl'), 'w', 'utf-8')
    pid2abstract = {}
    for fname in filenames:
        with codecs.open(os.path.join(in_abs_path, fname), 'r', 'utf-8') as fp:
            file_lines = fp.readlines()
        title = file_lines[0].strip()
        abs_text = ' '.join([s.strip() for s in file_lines[1:]])
        abs_sentences = scispacy_model(abs_text,
                                       disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                'lemmatizer', 'parser', 'ner'])
        abs_sentences = [sent.text for sent in abs_sentences.sents]
        if title and len(abs_sentences) > 0:
            pmid = fname[7:-4]  # filenames are like: PubMed-25010440.txt
            doc_dict = {'title': title, 'abstract': abs_sentences, 'paper_id': pmid}
            pid2abstract[pmid] = doc_dict
            out_file.write(json.dumps(doc_dict)+'\n')
    print('Docs with data: {:d}'.format(len(pid2abstract)))
    print('Wrote: {:s}'.format(out_file.name))
    out_file.close()
    
    with codecs.open(os.path.join(in_path, 'RELISH_v1_ann.json'), 'r', 'utf-8') as fp:
        ann_dicts = json.load(fp)

    query_meta_file = codecs.open(os.path.join(out_path, 'relish-queries-release.csv'), 'w', 'utf-8')
    query_meta_csv = csv.DictWriter(query_meta_file, extrasaction='ignore',
                                    fieldnames=['paper_id', 'title'])
    query_meta_csv.writeheader()
    
    query_pmids = []
    num_cands_perq = []
    relevant_num_cands_perq = []
    partial_num_cands_perq = []
    irrelevant_num_cands_perq = []
    qpmid2cands = {}
    for ann_dict in ann_dicts:
        qpid = ann_dict['pmid']
        query_pmids.append(ann_dict['pmid'])
        if qpid not in pid2abstract:
            continue
        cands = []
        relevances = []
        for cpid in ann_dict['response']['relevant']:
            if cpid not in pid2abstract:
                continue
            cands.append(cpid)
            relevances.append(2)
        relevant_num_cands_perq.append(len(ann_dict['response']['relevant']))
        for cpid in ann_dict['response']['partial']:
            if cpid not in pid2abstract:
                continue
            cands.append(cpid)
            relevances.append(1)
        partial_num_cands_perq.append(len(ann_dict['response']['partial']))
        for cpid in ann_dict['response']['irrelevant']:
            if cpid not in pid2abstract:
                continue
            cands.append(cpid)
            relevances.append(0)
        irrelevant_num_cands_perq.append(len(ann_dict['response']['irrelevant']))
        if cands:
            qpmid2cands[qpid] = {'cands': cands, 'relevance_adju': relevances}
            query_meta_csv.writerow({'title': pid2abstract[qpid]['title'], 'paper_id': qpid})
        # Check that there arent papers with multiple ratings.
        assert(len(set(cands)) == len(cands))
        num_cands_perq.append(len(cands))
    print('Query PMIDs: {:d}'.format(len(query_pmids)))
    cand_summ = pd.DataFrame(num_cands_perq).describe()
    print('Candidates per query: {:}'.format(cand_summ))
    cand_summ = pd.DataFrame(relevant_num_cands_perq).describe()
    print('Relevant candidates per query: {:}'.format(cand_summ))
    cand_summ = pd.DataFrame(partial_num_cands_perq).describe()
    print('Partial candidates per query: {:}'.format(cand_summ))
    cand_summ = pd.DataFrame(irrelevant_num_cands_perq).describe()
    print('Irrelevant candidates per query: {:}'.format(cand_summ))
    with codecs.open(os.path.join(out_path, 'test-pid2anns-relish.json'), 'w') as fp:
        json.dump(qpmid2cands, fp)
        print('Wrote: {:s}'.format(fp.name))
    print('Wrote: {:}'.format(query_meta_file.name))
    query_meta_file.close()
    
    
def pprint_graded_anns(data_path):
    """
    Given jsonl abstracts of the papers and the pid2anns-relish.json files print out for the
    query all the papers which are similar.
    :param data_path:
    :return:
    """
    sim2str = {
        0: 'Irrelevant (0)',
        1: 'Partial (+1)',
        2: 'Relevant (+2)'
    }
    pid2abstract = {}
    with codecs.open(os.path.join(data_path, 'abstracts-relish.jsonl'), 'r', 'utf-8') as fp:
        for line in fp:
            jsond = json.loads(line.strip())
            pid2abstract[jsond['paper_id']] = jsond
    with codecs.open(os.path.join(data_path, 'test-pid2anns-relish.json'), 'r', 'utf-8') as fp:
        pid2anns = json.load(fp)
    
    for qpid in pid2anns.keys():
        print('Processing: {:}'.format(qpid))
        resfile = codecs.open(os.path.join(data_path, 'readable_annotations/{:}.txt'.format(qpid)), 'w', 'utf-8')
        cand_pids = pid2anns[qpid]['cands']
        relevances = pid2anns[qpid]['relevance_adju']
        cand2rel = dict([(c, r) for c, r in zip(cand_pids, relevances)])
        # Write query.
        try:
            qtitle = pid2abstract[qpid]['title']
            qabs = '\n'.join(pid2abstract[qpid]['abstract'])
        except KeyError:
            print('Missing query: {:}'.format(qpid))
            continue
        resfile.write('======================================================================\n')
        resfile.write('paper_id: {:s}\n'.format(qpid))
        resfile.write('TITLE: {:s}\n'.format(qtitle))
        resfile.write('ABSTRACT: {:s}\n'.format(qabs))
        for cpid in sorted(cand2rel, key=cand2rel.get, reverse=True):
            resfile.write('===================================\n')
            try:
                ntitle = pid2abstract[cpid]['title']
                nabs = '\n'.join(pid2abstract[cpid]['abstract'])
            except KeyError:
                print('Missing candidate: {:s}'.format(cpid))
                continue
            resfile.write('paper_id: {:s}\n'.format(cpid))
            resfile.write('relevance: {:}\n'.format(sim2str[cand2rel[cpid]]))
            resfile.write('TITLE: {:s}\n'.format(ntitle))
            resfile.write('ABSTRACT: {:s}\n\n'.format(nabs))
        resfile.close()


def setup_splits(in_path, out_path):
    """
    Read in queries release file and write out half the queries as
    dev and the rest as test. Make the splits at the level of topics.
    """
    random.seed(582)
    with codecs.open(os.path.join(in_path, 'relish-queries-release.csv'), 'r', 'utf-8') as fp:
        csv_reader = csv.DictReader(fp)
        query_pids = []
        for row in csv_reader:
            query_pids.append(row['paper_id'])
    
    random.shuffle(query_pids)
    
    dev = query_pids[:len(query_pids)//2]
    test = query_pids[len(query_pids)//2:]
    eval_splits = {'dev': dev, 'test': test}
    print(f'dev_pids: {len(dev)}; test_pids: {len(test)}')
    
    with codecs.open(os.path.join(out_path, 'relish-evaluation_splits.json'), 'w', 'utf-8') as fp:
        json.dump(eval_splits, fp)
        print('Wrote: {:s}'.format(fp.name))
        

if __name__ == '__main__':
    # annotation_pmids(in_path='/iesl/canvas/smysore/facetid_apps/datasets_raw/relish_v1/')
    # ann_stats2json(in_abs_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/relish/'
    #                            'neves_collected/RELISH-DB/texts',
    #                in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/relish',
    #                out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/relish')
    # pprint_graded_anns(data_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/relish/')
    setup_splits(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/relish',
                 out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/relish')
