"""
Process the TREC-COVID dataset into a form i use.
"""
import os
import codecs
import json
import collections
import random
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import csv
import spacy

import data_utils as du

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def topics2json(in_path, out_path):
    """
    Convert the xml topics file to a json file.
    """
    in_fname = os.path.join(in_path, 'topics-rnd5.xml')
    doc_tree = ET.parse(in_fname)
    doc_root = doc_tree.getroot()
    topic2meta = {}
    for child in doc_root.iter():
        if child.tag == 'topic':
            number = child.attrib['number']
            d = {}
            for s in child.iter():
                if s.tag in {'query', 'question', 'narrative'}:
                    d[s.tag] = s.text
            topic2meta[number] = d
    
    with codecs.open(os.path.join(out_path, 'topics-rnd5.json'), 'w', 'utf-8') as fp:
        json.dump(topic2meta, fp, indent=2)


def print_relevances(in_path, out_path):
    """
    - Read in qrels.
    - Read in metadata.
    - For every topics relevant articles create a pool consisting of not relevant ones
        from every other topic.
    """
    # Read in abstracts.
    meta_fname = os.path.join(in_path, 'metadata.csv')
    abstracts_meta = pd.read_csv(meta_fname, delimiter=',', error_bad_lines=False)

    # Read in qrels file.
    qrel_file = codecs.open(os.path.join(in_path, 'qrels-covid_d5_j0.5-5.txt'), 'r', 'utf-8')
    topic2judgement_pool = collections.defaultdict(list)
    for qrel_line in qrel_file:
        parts = qrel_line.strip().split()
        topic_id, jround, doc_id, judgement = parts[0], parts[1], parts[2], parts[3]
        topic2judgement_pool[topic_id].append((doc_id, judgement))
    topic2judgement_pool = dict(topic2judgement_pool)
    for topic_id in topic2judgement_pool:
        topic2judgement_pool[topic_id] = dict(topic2judgement_pool[topic_id])
        
    # Read in topics json.
    with codecs.open(os.path.join(in_path, 'topics-rnd5.json'), 'r', 'utf-8') as fp:
        topics = json.load(fp)
    
    # Print out relevance ratings for the original query for examination.
    topic_qrels_path = os.path.join(in_path, 'readable_topic_qrels')
    du.create_dir(topic_qrels_path)
    
    for topic_id in range(1, 51, 1):
        # Print out distribution of relevances.
        judgement2cand = collections.defaultdict(list)
        for cand_did, rel in topic2judgement_pool[str(topic_id)].items():
            judgement2cand[rel].append(cand_did)
        j2len = []
        for rel in judgement2cand:
            j2len.append((rel, len(judgement2cand[rel])))
        j2len = dict(j2len)
        print('topic: {:d}; relevances: {:}'.format(topic_id, j2len))
        query, question, narrative = topics[str(topic_id)]['query'], \
                                     topics[str(topic_id)]['question'], \
                                     topics[str(topic_id)]['narrative']
        # Print out a handful of documents from each relevance level.
        outf = codecs.open(os.path.join(topic_qrels_path, f'{topic_id}-readable.txt'), 'w', 'utf-8')
        print(f'topic_id: {topic_id}', file=outf)
        print(f'query: {query}', file=outf)
        print(f'question: {question}', file=outf)
        print(f'narrative: {narrative}\n\n', file=outf)
        for rel, cands in sorted(judgement2cand.items(), key=lambda i: i[0], reverse=True):
            out_cands = random.sample(cands, min(25, len(cands)))
            for doc_id in out_cands:
                doc_row = abstracts_meta.loc[abstracts_meta['cord_uid'] == doc_id]
                doc_row = doc_row.to_dict()
                print(f'cord_uid: {doc_id}', file=outf)
                print(f'relevance: {rel}', file=outf)
                try:
                    title = list(doc_row['title'].values())[0]
                except IndexError:
                    title = None
                try:
                    abstract = list(doc_row['abstract'].values())[0]
                except IndexError:
                    abstract = None
                print(f"Title: {title}", file=outf)
                print(f"Abstract:\n {abstract}", file=outf)
                print('====================', file=outf)


def get_qbe_pools(in_path, out_path):
    """
    - Read in qrels.
    - Get abstracts
    """
    random.seed(472945)
    # Read in abstracts.
    meta_fname = os.path.join(in_path, 'metadata-2021-06-21.csv')
    rel_meta = pd.read_csv(meta_fname, delimiter=',', error_bad_lines=False)

    # Read in topics json.
    with codecs.open(os.path.join(in_path, 'topics-rnd5.json'), 'r', 'utf-8') as fp:
        topics = json.load(fp)
        
    # Read in only the top relevant docs.
    qrel_file = codecs.open(os.path.join(in_path, 'qrels-covid_d5_j0.5-5.txt'), 'r', 'utf-8')
    topic2relevant_pool = collections.defaultdict(list)
    docid2reltopic = collections.defaultdict(list)
    for qrel_line in qrel_file:
        parts = qrel_line.strip().split()
        topic_id, jround, doc_id, judgement = parts[0], parts[1], parts[2].strip(), parts[3]
        if judgement == '2':
            topic2relevant_pool[topic_id].append(doc_id)
            docid2reltopic[doc_id].append(topic_id)
    
    num_relevant = []
    for topic, reldocs in topic2relevant_pool.items():
        num_relevant.append(len(reldocs))
    summary = pd.DataFrame(num_relevant).describe()
    print('Relevant docs: {:}'.format(summary))
    
    all_docs = [item for sublist in topic2relevant_pool.values() for item in sublist]
    all_docs_uniq = list(set(all_docs))
    print('Corpus size: {:d}; Unique corpus size: {:d}'.format(len(all_docs), len(all_docs_uniq)))
    
    # Read in abstracts of the papers which are relevant
    abstract_jsonl = codecs.open(os.path.join(out_path, 'abstracts-treccovid.jsonl'), 'w', 'utf-8')
    pid2abstract = {}
    abstract_not_obtained = 0
    docs_with_data = set()
    useful_subset = rel_meta.loc[rel_meta['cord_uid'].isin(all_docs_uniq)]
    print('Docs found in metadata: {:}'.format(useful_subset.shape))
    for idx, doc_row in useful_subset.iterrows():
        doc_id = doc_row['cord_uid']
        # doc_row = rel_meta.loc[rel_meta['cord_uid'] == doc_id]
        # doc_row = doc_row.to_dict()
        try:
            title = doc_row['title']
            abs_text = doc_row['abstract']
        except IndexError:
            title = None
            abs_text = None
        if isinstance(title, str) and isinstance(abs_text, str) and (doc_id not in docs_with_data):
            date = doc_row['publish_time']
            year = date.split('-')[0] if isinstance(date, str) else None
            authors = doc_row['authors']
            if not isinstance(authors, str):
                authors = None
            try:
                abstract_sents = scispacy_model(abs_text,
                                                disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                         'lemmatizer', 'parser', 'ner'])
            except TypeError:
                print(doc_id)
                print(abs_text)
                sys.exit()
            abs_sentences = [sent.text for sent in abstract_sents.sents]
            doc_rel_topics = docid2reltopic[doc_id]
            narratives = [topics[t] for t in doc_rel_topics]
            docd = {
                'paper_id': doc_id,
                'title': title.strip(),
                'abstract': abs_sentences,
                'metadata': {'year': year, 'authors': authors},
                'topic_ids': '-'.join(doc_rel_topics),
                'topic_narratives': narratives
            }
            pid2abstract[doc_id] = docd
            abstract_jsonl.write(json.dumps(docd)+'\n')
            docs_with_data.add(doc_id)
        else:
            abstract_not_obtained += 1
    print('Docs without abstract/titles: {:d}'.format(abstract_not_obtained))
    print('Wrote: {:}'.format(abstract_jsonl.name))
    abstract_jsonl.close()
    assert(len(docs_with_data) == len(pid2abstract))
    
    # Build relevance annotation file;
    # Only do this for docs which have abstracts present.
    topic2relevant_pool_present = collections.defaultdict(list)
    for topicid, pool in topic2relevant_pool.items():
        for pid in pool:
            if pid in docs_with_data:
                topic2relevant_pool_present[topicid].append(pid)
    print('Topics with valid docs: {:d}'.format(len(topic2relevant_pool_present)))
    # Only use queries which are relevant for a single topic.
    multi_rel_docs = []
    for doc_id, reltopics in docid2reltopic.items():
        if len(reltopics) > 1:
            multi_rel_docs.append(doc_id)
    print('Docs relevant for multiple topics: {:d}'.format(len(multi_rel_docs)))
    qpid2anns = {}
    all_qbe_qpids = []
    num_cands = []
    # Go over the 50 topics and selectt 50 docs at random to act as queries
    # and get positives and negatives wrt these.
    for topicid, relpool in sorted(topic2relevant_pool_present.items(), key=lambda i: len(i[1])):
        tqpids = []
        random.shuffle(relpool)
        for tpid in relpool:
            # Get per topic queries such that they're unique across topics;
            # exclude docs relevant to multiple topics and there are atmost 50 queries per query
            if (tpid not in all_qbe_qpids) and (tpid not in multi_rel_docs) and (len(tqpids) < 50):
                tqpids.append(tpid)
        print(f'topic: {topicid}; QBE queries: {len(tqpids)}')
        all_qbe_qpids.extend(tqpids)
        for qpid in tqpids:
            pos_cand_pool = [pid for pid in relpool if pid != qpid]
            pool_rels = [1]*len(pos_cand_pool)
            # All docs relevant to other topics are negatives. -
            # if there are docs relevant to multiple topics those are not included as negatives.
            neg_cand_pool = list(set.difference(set(docs_with_data), set(relpool)))
            negpool_rels = [0]*len(neg_cand_pool)
            cands = pos_cand_pool + neg_cand_pool
            rels = pool_rels + negpool_rels
            assert(len(cands) == len(rels))
            qpid2anns[qpid] = {'cands': cands, 'relevance_adju': rels}
            num_cands.append(len(cands))
    print('Number of QBE queries: {:d}; unique QBE queries: {:d}'.
          format(len(all_qbe_qpids), len(set(all_qbe_qpids))))
    csum = pd.DataFrame(num_cands).describe()
    print('Number of candidates per QBE query: {:}'.format(csum))
    with codecs.open(os.path.join(out_path, 'test-pid2anns-treccovid.json'), 'w', 'utf-8') as fp:
        json.dump(qpid2anns, fp)
        print('Wrote: {:}'.format(fp.name))
    # Build queries release file.
    query_meta_file = codecs.open(os.path.join(out_path, 'treccovid-queries-release.csv'), 'w', 'utf-8')
    query_meta_csv = csv.DictWriter(query_meta_file, extrasaction='ignore',
                                    fieldnames=['paper_id', 'title', 'year', 'topic_ids'])
    query_meta_csv.writeheader()
    for qpid in all_qbe_qpids:
        md = {'paper_id': qpid,
              'title': pid2abstract[qpid]['title'],
              'year': pid2abstract[qpid]['metadata']['year'],
              'topic_ids': pid2abstract[qpid]['topic_ids']}
        query_meta_csv.writerow(md)
    print('Wrote: {:}'.format(query_meta_file.name))
    query_meta_file.close()
    
    
def setup_splits(in_path, out_path):
    """
    Read in queries release file and write out half the queries as
    dev and the rest as test. Make the splits at the level of topics.
    """
    random.seed(582)
    with codecs.open(os.path.join(in_path, 'treccovid-queries-release.csv'), 'r', 'utf-8') as fp:
        csv_reader = csv.DictReader(fp)
        topic2pids = collections.defaultdict(list)
        for row in csv_reader:
            topic2pids[row['topic_ids']].append(row['paper_id'])
        
    topic_ids = list(topic2pids.keys())
    topic_ids.sort()
    
    random.shuffle(topic_ids)
    
    dev_topics = topic_ids[:len(topic_ids)//2]
    dev = [topic2pids[tid] for tid in dev_topics]
    dev = [item for sublist in dev for item in sublist]
    test_topics = topic_ids[len(topic_ids)//2:]
    test = [topic2pids[tid] for tid in test_topics]
    test = [item for sublist in test for item in sublist]
    eval_splits = {'dev': dev, 'test': test}
    print(f'dev_topics: {len(dev_topics)}; test_topics: {len(test_topics)}')
    print(f'dev_pids: {len(dev)}; test_pids: {len(test)}')
    
    with codecs.open(os.path.join(out_path, 'treccovid-evaluation_splits.json'), 'w', 'utf-8') as fp:
        json.dump(eval_splits, fp)
        print('Wrote: {:s}'.format(fp.name))
    

if __name__ == '__main__':
    # topics2json(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid',
    #             out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid')
    
    # print_relevances(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid',
    #                  out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid')
    
    # get_qbe_pools(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid',
    #               out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid')
    
    setup_splits(in_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid',
                 out_path='/iesl/canvas/smysore/2021-ai2-scisim/datasets_raw/trec-covid')
