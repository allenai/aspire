"""
Functions to work with co-citations in each area.
"""
import os
import random
import math
import argparse
import time
import collections
import itertools
import re
import pprint
import pickle
import codecs, json
import pandas as pd
import torch
import numpy as np
# import spacy
from sentence_transformers import SentenceTransformer, models

import data_utils as du
import pp_settings as pps

# scispacy_model = spacy.load("en_core_sci_sm")
# scispacy_model.add_pipe('sentencizer')


class AbsSentenceStream:
    """
    Given a list of pids,  returns their sentences.
    """
    
    def __init__(self, in_pids, pid2abstract):
        """
        :param in_pids:
        :param pid2abstract:
        """
        self.in_pids = in_pids
        self.pid2abstract = pid2abstract
        self.num_sents = self.count_sents()
    
    def __len__(self):
        return self.num_sents

    def count_sents(self):
        nsents = 0
        for pid in self.in_pids:
            doc = self.pid2abstract[pid]['abstract']
            nsents += len(doc)
        return nsents
    
    def __iter__(self):
        return self.next()
    
    def next(self):
        # In each loop iteration return one example.
        for pid in self.in_pids:
            doc = self.pid2abstract[pid]['abstract']
            for sent in doc:
                yield sent


class ContextSentenceStream:
    """
    Given a list of pids,  returns their sentences.
    """
    
    def __init__(self, listofcontexts):
        """
        :param listofcontexts: list(list(tuple(pid, sent)))
        """
        self.listofcontexts = listofcontexts
        self.num_sents = self.count_sents()
    
    def __len__(self):
        return self.num_sents
    
    def count_sents(self):
        nsents = 0
        for clist in self.listofcontexts:
            nsents += len(clist)
        return nsents
    
    def __iter__(self):
        return self.next()
    
    def next(self):
        # In each loop iteration return one example.
        for clist in self.listofcontexts:
            for c in clist:
                yield c[1]
        

def filter_cocitation_papers(run_path, dataset):
    """
    Read in the absfilt co-cotations and filter out co-citations using:
    - the number of cocited papers.
    - the number of tokens in the citation context.
    - if the citation context was supriously tagged as a citation context:
        - The heuristic for this is when the sentence doesnt contain any [] or ().
          This is  more important in biomed papers than in compsci papers.
    This is used to train the abstract level similarity models.
    """
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed',
        's2orcmatsci': 'matsci'
    }
    area = dataset2area[dataset]
    with open(os.path.join(run_path, f'cocitpids2contexts-{area}-absfilt.pickle'), 'rb') as fp:
        cocitpids2contexts = pickle.load(fp)

    # Filter out noise.
    cocitedpids2contexts_filt = {}
    sc_copy_count = 0
    for cocitpids, contexts in cocitpids2contexts.items():
        if len(cocitpids) > 3:
            continue
        else:
            # Sometimes the contexts are exact copies but from diff papers.
            # Get rid of these.
            con2pids = collections.defaultdict(list)
            for sc in contexts:
                # Sometimes they differ only by the inline citation numbers, replace those.
                sc_no_nums = re.sub(r'\d', '', sc[1])
                con2pids[sc_no_nums].append(sc)
            if len(con2pids) < len(contexts):
                sc_copy_count += 1
            uniq_scons = []
            for norm_con, contextt in con2pids.items():
                uniq_scons.append(contextt[0])
            fcons = []
            citing_pids = set()
            for sc in uniq_scons:
                # If the same paper is making the co-citation multiple times
                # only use the first of the co-citations. Multiple by the same citing
                # paper count as a single co-citation.
                if sc[0] in citing_pids:
                    continue
                # Filter context by length.
                if len(sc[1].split()) > 60 or len(sc[1].split()) < 5:
                    continue
                # Filter noisey citation contexts.
                elif ("(" not in sc[1] and ")" not in sc[1]) and ("[" not in sc[1] and "]" not in sc[1]):
                    continue
                else:
                    fcons.append(sc)
                # Update pids only if the sentence was used.
                citing_pids.add(sc[0])
            if len(fcons) > 0:
                cocitedpids2contexts_filt[cocitpids] = fcons

    # Write out filtered co-citations and their stats.
    with codecs.open(os.path.join(run_path, f'cocitpids2contexts-{area}-absnoisefilt.pickle'), 'wb') as fp:
        pickle.dump(cocitedpids2contexts_filt, fp)
        print(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(run_path, f'cocitpids2contexts-{area}-absnoisefilt.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(cocitedpids2contexts_filt.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        print(f'Wrote: {fp.name}')
    num_citcons = []
    example_count = 0  # The approximate number of triples which will be generated as training data.
    for cocitpids, citcontexts in cocitedpids2contexts_filt.items():
        num_citcons.append(len(citcontexts))
        if len(cocitpids) == 2:
            example_count += 1
        elif len(cocitpids) == 3:
            example_count += 3
    all_summ = pd.DataFrame(num_citcons).describe()
    print('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))
    print(f'Copies of co-citation context: {sc_copy_count}')
    print(f'Approximate number of possible triple examples: {example_count}')


def filter_cocitation_sentences(run_path, dataset):
    """
    Generate data to train sentence level "paraphrasing" models like SentBERT.
    For papers which are cocited cited more than once:
    - the number of tokens in the citation context.
    - if the citation context was supriously tagged as a citation context:
        - The heuristic for this is when the sentence doesnt contain any [] or ().
          This is  more important in biomed papers than in compsci papers.
    """
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed',
        's2orcmatsci': 'matsci'
    }
    area = dataset2area[dataset]
    with open(os.path.join(run_path, f'cocitpids2contexts-{area}-absfilt.pickle'), 'rb') as fp:
        cocitpids2contexts = pickle.load(fp)

    # Gather sentences which are roughly paraphrases.
    cocitedpids2contexts_filt = {}
    sc_copy_count = 0
    for cocitpids, contexts in cocitpids2contexts.items():
        if len(contexts) < 2:
            continue
        else:
            # Sometimes the contexts are exact copies but from diff papers.
            # Get rid of these.
            con2pids = collections.defaultdict(list)
            for sc in contexts:
                # Sometimes they differ only by the inline citation numbers, replace those.
                sc_no_nums = re.sub(r'\d', '', sc[1])
                con2pids[sc_no_nums].append(sc)
            if len(con2pids) < len(contexts):
                sc_copy_count += 1
            uniq_scons = []
            for norm_con, contextt in con2pids.items():
                uniq_scons.append(contextt[0])
            fcons = []
            citing_pids = set()
            for sc in uniq_scons:
                # If the same paper is making the co-citation multiple times
                # only use the first of the co-citations. Multiple by the same citing
                # paper count as a single co-citation.
                if sc[0] in citing_pids:
                    continue
                # Filter context by length.
                if len(sc[1].split()) > 60 or len(sc[1].split()) < 5:
                    continue
                # Filter noisey citation contexts.
                elif ("(" not in sc[1] and ")" not in sc[1]) and ("[" not in sc[1] and "]" not in sc[1]):
                    continue
                else:
                    fcons.append(sc)
                # Update pids only if the sentence was used.
                citing_pids.add(sc[0])
            if len(fcons) > 1:
                cocitedpids2contexts_filt[cocitpids] = fcons

    # Write out filtered co-citations and their stats.
    with codecs.open(os.path.join(run_path, f'cocitpids2contexts-{area}-sentfilt.pickle'), 'wb') as fp:
        pickle.dump(cocitedpids2contexts_filt, fp)
        print(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(run_path, f'cocitpids2contexts-{area}-sentfilt.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(cocitedpids2contexts_filt.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        print(f'Wrote: {fp.name}')
    num_cocited_pids = []
    num_citcons = []
    example_count = 0
    for cocitpids, citcontexts in cocitedpids2contexts_filt.items():
        num_cocited_pids.append(len(cocitpids))
        num_cons = len(citcontexts)
        num_citcons.append(num_cons)
        ex = math.factorial(num_cons)/(math.factorial(2)*math.factorial(num_cons-2))
        example_count += ex
    all_summ = pd.DataFrame(num_cocited_pids).describe()
    print('Papers co-cited together:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_cocited_pids)))
    all_summ = pd.DataFrame(num_citcons).describe()
    print('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))
    print(f'Copies of co-citation context: {sc_copy_count}')
    print(f'Approximate number of possible triple examples: {example_count}')


def generate_examples_sent_rand(in_path, out_path, dataset):
    """
    Assumes random (in-batch) negatives are used and only generates pair
    examples of query/anchor and positive.
    - Generate negative sentences for the dev set so its a frozen dev set.
    """
    random.seed(57395)
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed'
    }
    area = dataset2area[dataset]
    with codecs.open(os.path.join(in_path, f'cocitpids2contexts-{area}-sentfilt.pickle'), 'rb') as fp:
        cocitedpids2contexts = pickle.load(fp)
        print(f'Read: {fp.name}')
    
    all_cocits = list(cocitedpids2contexts.keys())
    random.shuffle(all_cocits)
    random.shuffle(all_cocits)
    total_copids = len(all_cocits)
    train_copids, dev_copids = all_cocits[:int(0.8*total_copids)], all_cocits[int(0.8*total_copids):]
    print(f'cocited pid sets; train: {len(train_copids)}; dev: {len(dev_copids)}')
    
    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-coppsent.jsonl'), 'w', 'utf-8')
        out_examples = 0
        for cocitedpids in split_copids:
            contexts = cocitedpids2contexts[cocitedpids]
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(contexts)), 2)
            for idxs in cidxs:
                anchor_context = contexts[idxs[0]]
                pos_context = contexts[idxs[1]]
                out_ex = {
                    'citing_pids': (anchor_context[0], pos_context[0]),
                    'cited_pids': cocitedpids,
                    'query': anchor_context[1],
                    'pos_context': pos_context[1]
                }
                # Of its dev also add a random negative context.
                if split_str == 'dev':
                    neg_copids = random.choice(split_copids)
                    neg_contexts = cocitedpids2contexts[neg_copids]
                    neg_context = random.choice(neg_contexts)
                    out_ex['neg_context'] = neg_context[1]
                out_ex_file.write(json.dumps(out_ex)+'\n')
                out_examples += 1
                if out_examples % 200000 == 0:
                    print(f'{split_str}; {out_examples}')
        print(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        print(f'Number of examples: {out_examples}')


def generate_examples_ict(in_path, out_path, dataset):
    """
    Assumes random (in-batch) negatives are used and only generates pair
    examples of sentence and abstract context.
    """
    random.seed(6036)
    if dataset == 's2orccompsci':
        area, num_abs, perabssentex = 'compsci', 1479197, 2
    elif dataset == 's2orcbiomed':
        area, num_abs, perabssentex = 'biomed', 10602028, 1
    # Use a shuffled jsonl file so its not ordered by batch number or something.
    in_abs_file = codecs.open(os.path.join(in_path, f'abstracts-{dataset}-shuf.jsonl'), 'r', 'utf-8')
    print(f'Reading: {in_abs_file.name}')
    
    num_train_abs, num_dev_abs = int(0.8*num_abs), int(0.2*num_abs)
    print(f'abstracts; train: {num_train_abs}; dev: {num_dev_abs}')
    
    ninty_plexoverlap = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    out_ex_file = codecs.open(os.path.join(out_path, 'train-ictsent.jsonl'), 'w', 'utf-8')
    out_examples = 0
    out_abs = 0
    split_str = 'train'
    for abs_line in in_abs_file:
        abs_json = json.loads(abs_line.strip())
        abs_sents = abs_json['abstract']
        query_sents = random.sample(abs_sents, perabssentex)
        for qsent in query_sents:
            # 90% of the time this will be 1 and qsent will be redacted else its present.
            lex_overlap = random.sample(ninty_plexoverlap, 1)
            if lex_overlap[0]:
                pos_context = ' '.join([s for s in abs_sents if s != qsent])
            else:
                pos_context = ' '.join(abs_sents)
            out_ex = {
                'paper_id': abs_json['paper_id'],
                'query': qsent,
                'pos_context': pos_context
            }
            out_ex_file.write(json.dumps(out_ex)+'\n')
            out_examples += 1
            if out_examples % 200000 == 0:
                print(f'{split_str}; {out_examples}')
        out_abs += 1
        if out_abs == num_train_abs:
            print(f'Wrote: {out_ex_file.name}')
            out_ex_file.close()
            print(f'Number of examples: {out_examples}')
            out_examples = 0
            out_abs = 0
            split_str = 'dev'
            out_ex_file = codecs.open(os.path.join(out_path, 'dev-ictsent.jsonl'), 'w', 'utf-8')
    # For the dev set.
    print(f'Wrote: {out_ex_file.name}')
    out_ex_file.close()
    print(f'Number of examples: {out_examples}')


def generate_examples_aligned_cocitabs_rand(in_path, out_path, dataset, alignment_model, trained_model_path=None):
    """
    Assumes random (in-batch) negatives are used and only generates pair
    examples of query/anchor and positive for co-cited abstracts.
    - Also generate a alignment for the positive and negative based
    - Generate negatives for the dev set so its a frozen dev set.
    """
    train_size, dev_size = 1276820, 10000
    random.seed(69306)
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed'
    }
    area = dataset2area[dataset]
    with codecs.open(os.path.join(in_path, f'cocitpids2contexts-{area}-absnoisefilt.pickle'), 'rb') as fp:
        cocitedpids2contexts = pickle.load(fp)
        print(f'Read: {fp.name}')
    
    with codecs.open(os.path.join(in_path, f'abstracts-s2orc{area}.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
        all_abs_pids = list(pid2abstract.keys())
        print(f'Read: {fp.name}')

    if alignment_model in {'cosentbert'}:
        outfname_suffix = 'cocitabsalign'
        word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                                  max_seq_length=512)
        trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
        word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        sent_alignment_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if alignment_model in {'sbmpnet1B'}:
        outfname_suffix = 'cocitabsalign-sb1b'
        sent_alignment_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    elif alignment_model in {'specter'}:
        outfname_suffix = 'cocitabsalign-spec'
        word_embedding_model = models.Transformer('allenai/specter', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sent_alignment_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
    all_cocits = list(cocitedpids2contexts.keys())
    random.shuffle(all_cocits)
    random.shuffle(all_cocits)
    total_copids = len(all_cocits)
    train_copids, dev_copids = all_cocits[:int(0.8*total_copids)], all_cocits[int(0.8*total_copids):]
    print(f'cocited pid sets; train: {len(train_copids)}; dev: {len(dev_copids)}')
    
    all_contexts = []
    all_pids = set()
    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        split_examples = 0
        for cocitedpids in split_copids:
            contexts = cocitedpids2contexts[cocitedpids]
            # Sample at most 10 context sentences at random to use for supervision.
            out_contexts = random.sample(contexts, min(10, len(contexts)))
            all_contexts.append(out_contexts)
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(cocitedpids)), 2)
            all_pids.update(cocitedpids)
            split_examples += len(list(cidxs))
            if split_str == 'train' and split_examples > train_size:
                break
            elif split_str == 'dev' and split_examples > dev_size:
                break
    all_pids = list(all_pids)
    print(f'Number of contexts: {len(all_contexts)}; Number of unique abstracts: {len(all_pids)}')
    context_stream = ContextSentenceStream(listofcontexts=all_contexts)
    abstract_stream = AbsSentenceStream(in_pids=all_pids, pid2abstract=pid2abstract)
    # Encode documents.
    pool = sent_alignment_model.start_multi_process_pool()
    # Compute the embeddings using the multi-process pool
    start = time.time()
    all_context_reps = sent_alignment_model.encode_multi_process(context_stream, pool)
    print(f"Context reps shape: {all_context_reps.shape}; Stream sents: {len(context_stream)}")
    all_abs_sent_reps = sent_alignment_model.encode_multi_process(abstract_stream, pool)
    print(f"Abs sent reps shape: {all_abs_sent_reps.shape}; Stream sents: {len(abstract_stream)}")
    # Optional: Stop the proccesses in the pool
    sent_alignment_model.stop_multi_process_pool(pool)
    print('Encoding took: {:.4f}s'.format(time.time()-start))
    # Go over the abstract reps and put them into a dict
    abs_reps_start_idx = 0
    pid2abs_reps = {}
    for pid in all_pids:
        num_sents = len(pid2abstract[pid]['abstract'])
        abs_reps = all_abs_sent_reps[abs_reps_start_idx:abs_reps_start_idx+num_sents, :]
        abs_reps_start_idx += num_sents
        pid2abs_reps[pid] = abs_reps
    
    # Now form examples.
    contextsi = 0
    context_reps_start_idx = 0
    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-{outfname_suffix}.jsonl'), 'w', 'utf-8')
        out_examples = 0
        num_context_sents = []
        for cocitedpids in split_copids:
            out_contexts = all_contexts[contextsi]
            context_sents = [cc[1] for cc in out_contexts]
            citing_pids = [cc[0] for cc in out_contexts]
            context_reps = all_context_reps[context_reps_start_idx: context_reps_start_idx+len(context_sents), :]
            context_reps_start_idx += len(context_sents)
            contextsi += 1
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(cocitedpids)), 2)
            for idxs in cidxs:
                anchor_pid = cocitedpids[idxs[0]]
                pos_pid = cocitedpids[idxs[1]]
                qabs_reps = pid2abs_reps[anchor_pid]
                posabs_reps = pid2abs_reps[pos_pid]
                cc2query_abs_sims = np.matmul(qabs_reps, context_reps.T)
                cc2query_idxs = np.unravel_index(cc2query_abs_sims.argmax(), cc2query_abs_sims.shape)
                cc2pos_abs_sims = np.matmul(posabs_reps, context_reps.T)
                cc2pos_idxs = np.unravel_index(cc2pos_abs_sims.argmax(), cc2pos_abs_sims.shape)
                abs2cc2abs_idx = (int(cc2query_idxs[0]), int(cc2pos_idxs[0]))
                q2pos_abs_sims = np.matmul(qabs_reps, posabs_reps.T)
                q2pos_idxs = np.unravel_index(q2pos_abs_sims.argmax(), q2pos_abs_sims.shape)
                abs2abs_idx = (int(q2pos_idxs[0]), int(q2pos_idxs[1]))
                anchor_abs = {'TITLE': pid2abstract[anchor_pid]['title'],
                              'ABSTRACT': pid2abstract[anchor_pid]['abstract']}
                pos_abs = {'TITLE': pid2abstract[pos_pid]['title'],
                           'ABSTRACT': pid2abstract[pos_pid]['abstract'],
                           'cc_align': abs2cc2abs_idx,
                           'abs_align': abs2abs_idx}
                out_ex = {
                    'citing_pids': citing_pids,
                    'cited_pids': cocitedpids,
                    'query': anchor_abs,
                    'pos_context': pos_abs,
                    'citing_contexts': context_sents
                }
                num_context_sents.append(len(citing_pids))
                # Of its dev also add a random negative context.
                if split_str == 'dev':
                    neg_pid = random.choice(all_abs_pids)
                    rand_anch_idx, rand_neg_idx = random.choice(range(len(pid2abstract[anchor_pid]['abstract']))), \
                                                  random.choice(range(len(pid2abstract[neg_pid]['abstract'])))
                    neg_cc_align = (rand_anch_idx, rand_neg_idx)
                    rand_anch_idx, rand_neg_idx = random.choice(range(len(pid2abstract[anchor_pid]['abstract']))), \
                                                  random.choice(range(len(pid2abstract[neg_pid]['abstract'])))
                    neg_abs_align = (rand_anch_idx, rand_neg_idx)
                    neg_abs = {'TITLE': pid2abstract[neg_pid]['title'],
                               'ABSTRACT': pid2abstract[neg_pid]['abstract'],
                               'cc_align': neg_cc_align, 'abs_align': neg_abs_align}
                    out_ex['neg_context'] = neg_abs
                out_ex_file.write(json.dumps(out_ex)+'\n')
                out_examples += 1
                if out_examples % 1000 == 0:
                    print(f'{split_str}; {out_examples}')
            # if out_examples > 1000:
            #     break
            # Do this only for 1.2m triples, then exit.
            if split_str == 'train' and out_examples > train_size:
                break
            elif split_str == 'dev' and out_examples > dev_size:
                break
        print(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        all_summ = pd.DataFrame(num_context_sents).describe()
        print('Number of cit contexts per triple: {:}'.format(all_summ))
        print(f'Number of examples: {out_examples}')


def generate_examples_cocitabs_rand(in_path, out_path, dataset):
    """
    Assumes random (in-batch) negatives are used and only generates pair
    examples of query/anchor and positive for co-cited abstracts.
    - Generate negatives for the dev set so its a frozen dev set.
    """
    random.seed(69306)
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed'
    }
    area = dataset2area[dataset]
    with codecs.open(os.path.join(in_path, f'cocitpids2contexts-{area}-absnoisefilt.pickle'), 'rb') as fp:
        cocitedpids2contexts = pickle.load(fp)
        print(f'Read: {fp.name}')

    with codecs.open(os.path.join(in_path, f'abstracts-s2orc{area}.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
        all_abs_pids = list(pid2abstract.keys())
        print(f'Read: {fp.name}')
    
    all_cocits = list(cocitedpids2contexts.keys())
    random.shuffle(all_cocits)
    random.shuffle(all_cocits)
    total_copids = len(all_cocits)
    train_copids, dev_copids = all_cocits[:int(0.8*total_copids)], all_cocits[int(0.8*total_copids):]
    print(f'cocited pid sets; train: {len(train_copids)}; dev: {len(dev_copids)}')

    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-cocitabs.jsonl'), 'w', 'utf-8')
        out_examples = 0
        num_context_sents = []
        for cocitedpids in split_copids:
            contexts = cocitedpids2contexts[cocitedpids]
            # Sample at most 10 context sentences at random to use for supervision.
            out_contexts = random.sample(contexts, min(10, len(contexts)))
            context_sents = [cc[1] for cc in out_contexts]
            citing_pids = [cc[0] for cc in out_contexts]
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(cocitedpids)), 2)
            for idxs in cidxs:
                anchor_pid = cocitedpids[idxs[0]]
                pos_pid = cocitedpids[idxs[1]]
                anchor_abs = {'TITLE': pid2abstract[anchor_pid]['title'],
                              'ABSTRACT': pid2abstract[anchor_pid]['abstract']}
                pos_abs = {'TITLE': pid2abstract[pos_pid]['title'],
                           'ABSTRACT': pid2abstract[pos_pid]['abstract']}
                out_ex = {
                    'citing_pids': citing_pids,
                    'cited_pids': cocitedpids,
                    'query': anchor_abs,
                    'pos_context': pos_abs,
                    'citing_contexts': context_sents
                }
                num_context_sents.append(len(citing_pids))
                # Of its dev also add a random negative context.
                if split_str == 'dev':
                    neg_pid = random.choice(all_abs_pids)
                    neg_abs = {'TITLE': pid2abstract[neg_pid]['title'],
                               'ABSTRACT': pid2abstract[neg_pid]['abstract']}
                    out_ex['neg_context'] = neg_abs
                out_ex_file.write(json.dumps(out_ex)+'\n')
                out_examples += 1
                if out_examples % 200000 == 0:
                    print(f'{split_str}; {out_examples}')
        print(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        all_summ = pd.DataFrame(num_context_sents).describe()
        print('Number of cit contexts per triple: {:}'.format(all_summ))
        print(f'Number of examples: {out_examples}')


def generate_examples_cocitabs_contexts_rand(in_path, out_path, dataset):
    """
    Assumes random (in-batch) negatives are used and only generates pair
    examples of query/anchor and positive for co-cited abstracts.
    - Bundles the co-citation context for the positive with the pos abstract.
    - Generate negatives for the dev set so its a frozen dev set.
        Additionally, generates negatives which are sampled from a valid co-cite
        set so they come with negative contexts.
    """
    train_size, dev_size = 1276820, 10000
    random.seed(69306)
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed'
    }
    area = dataset2area[dataset]
    with codecs.open(os.path.join(in_path, f'cocitpids2contexts-{area}-absnoisefilt.pickle'), 'rb') as fp:
        cocitedpids2contexts = pickle.load(fp)
        print(f'Read: {fp.name}')
    
    with codecs.open(os.path.join(in_path, f'abstracts-s2orc{area}.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
        print(f'Read: {fp.name}')
    
    all_cocits = list(cocitedpids2contexts.keys())
    random.shuffle(all_cocits)
    random.shuffle(all_cocits)
    total_copids = len(all_cocits)
    train_copids, dev_copids = all_cocits[:int(0.8*total_copids)], all_cocits[int(0.8*total_copids):]
    print(f'cocited pid sets; train: {len(train_copids)}; dev: {len(dev_copids)}')

    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-concocitabs-seq.jsonl'), 'w', 'utf-8')
        out_examples = 0
        num_context_sents = []
        for cocitedpids in split_copids:
            contexts = cocitedpids2contexts[cocitedpids]
            # Sample at most 10 context sentences at random to use for supervision.
            out_contexts = random.sample(contexts, min(10, len(contexts)))
            context_sents = [cc[1] for cc in out_contexts]
            citing_pids = [cc[0] for cc in out_contexts]
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(cocitedpids)), 2)
            for idxs in cidxs:
                anchor_pid = cocitedpids[idxs[0]]
                pos_pid = cocitedpids[idxs[1]]
                anchor_abs = {'TITLE': pid2abstract[anchor_pid]['title'],
                              'ABSTRACT': pid2abstract[anchor_pid]['abstract']}
                pos_abs = {'TITLE': pid2abstract[pos_pid]['title'],
                           'ABSTRACT': pid2abstract[pos_pid]['abstract'],
                           'citing_contexts': context_sents,
                           'citing_pids': citing_pids}
                out_ex = {
                    'cited_pids': cocitedpids,
                    'query': anchor_abs,
                    'pos_context': pos_abs
                }
                num_context_sents.append(len(citing_pids))
                # Of its dev also add a random negative context.
                if split_str == 'dev':
                    neg_cocit_pids = random.choice(all_cocits)
                    neg_contexts = cocitedpids2contexts[neg_cocit_pids]
                    neg_out_contexts = random.sample(neg_contexts, min(10, len(neg_contexts)))
                    neg_context_sents = [cc[1] for cc in neg_out_contexts]
                    neg_citing_pids = [cc[0] for cc in neg_out_contexts]
                    # Sample at most 10 context sentences at random to use for supervision.
                    neg_pid = random.choice(neg_cocit_pids)
                    neg_abs = {'TITLE': pid2abstract[neg_pid]['title'],
                               'ABSTRACT': pid2abstract[neg_pid]['abstract'],
                               'citing_contexts': neg_context_sents,
                               'citing_pids': neg_citing_pids}
                    out_ex['neg_context'] = neg_abs
                out_ex_file.write(json.dumps(out_ex)+'\n')
                out_examples += 1
                if out_examples % 200000 == 0:
                    print(f'{split_str}; {out_examples}')
            # if out_examples > 1000:
            #     break
            # Do this only for 1.2m triples, then exit.
            if split_str == 'train' and out_examples > train_size:
                break
            elif split_str == 'dev' and out_examples > dev_size:
                break
        print(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        all_summ = pd.DataFrame(num_context_sents).describe()
        print('Number of cit contexts per triple: {:}'.format(all_summ))
        print(f'Number of examples: {out_examples}')


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')
    
    # Filter for abstract level models.
    filter_cocit_papers = subparsers.add_parser('filt_cocit_papers')
    filter_cocit_papers.add_argument('--run_path', required=True,
                                     help='Directory with absfilt cocitation pickle file. '
                                          'Also where outputs are written.')
    filter_cocit_papers.add_argument('--dataset', required=True,
                                     choices=['s2orccompsci', 's2orcbiomed', 's2orcmatsci'],
                                     help='Files of area to process.')
    # Filter for sentence level models.
    filter_cocit_sents = subparsers.add_parser('filt_cocit_sents')
    filter_cocit_sents.add_argument('--run_path', required=True,
                                    help='Directory with absfilt cocitation pickle file. '
                                         'Also where outputs are written.')
    filter_cocit_sents.add_argument('--dataset', required=True,
                                    choices=['s2orccompsci', 's2orcbiomed', 's2orcmatsci'],
                                    help='Files of area to process.')
    # Write examples for sentence level models.
    write_example_sents = subparsers.add_parser('write_examples')
    write_example_sents.add_argument('--in_path', required=True,
                                     help='Directory with absfilt cocitation pickle file.')
    write_example_sents.add_argument('--out_path', required=True,
                                     help='Directory where outputs are written.')
    write_example_sents.add_argument('--model_path',
                                     help='Directory where trained sentence bert model is.')
    write_example_sents.add_argument('--model_name', choices=['cosentbert', 'specter', 'sbmpnet1B'],
                                     help='Model to use for getting alignments between abstracts.')
    write_example_sents.add_argument('--dataset', required=True,
                                     choices=['s2orccompsci', 's2orcbiomed', 'treccovid', 'relish'],
                                     help='Files of area to process.')
    write_example_sents.add_argument('--experiment', required=True,
                                     choices=['cosentbert', 'ictsentbert', 'cospecter',
                                              'labspecter', 'consentsimcse', 'sbalisentbienc', 'alisentbienc',
                                              'alisentbienccf'],
                                     help='Model writing examples for.')
    cl_args = parser.parse_args()
    
    if cl_args.subcommand == 'filt_cocit_papers':
        filter_cocitation_papers(run_path=cl_args.run_path, dataset=cl_args.dataset)
    elif cl_args.subcommand == 'filt_cocit_sents':
        filter_cocitation_sentences(run_path=cl_args.run_path, dataset=cl_args.dataset)
    elif cl_args.subcommand == 'write_examples':
        if cl_args.experiment in {'cosentbert'}:
            generate_examples_sent_rand(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                        dataset=cl_args.dataset)
        elif cl_args.experiment in {'ictsentbert'}:
            generate_examples_ict(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                  dataset=cl_args.dataset)
        elif cl_args.experiment in {'cospecter'}:
            generate_examples_cocitabs_rand(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                            dataset=cl_args.dataset)
        elif cl_args.experiment in {'sbalisentbienc'}:
            generate_examples_aligned_cocitabs_rand(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                                    dataset=cl_args.dataset, trained_model_path=cl_args.model_path,
                                                    alignment_model=cl_args.model_name)


if __name__ == '__main__':
    main()
