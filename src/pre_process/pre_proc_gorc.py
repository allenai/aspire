"""
Explore the GORC corpus for corpora included and such.
"""
import sys
import os
import random
import ast
import argparse
import time
import gzip
import multiprocessing as mp
import collections
import pprint
import pickle
import codecs, json
import csv
import pandas as pd
from nltk import tokenize
import spacy

import data_utils as du
import pp_settings as pps

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def filter_for_fulltext(args):
    """
    Open the metadata file, and return papaers of specific area.
    Not checking for hosting sites, only Microsoft Academic Graph Field of Study.
    :param in_fname: string.
    :param filter_columns: will always be None here. Just preserving the same function
        prototype to not change the filter_by_hostingservice function.
    :return:
    """
    in_fname, filter_columns = args
    meta_csv = pd.read_csv(in_fname, delimiter='\t', error_bad_lines=False,
                           engine='python', quoting=csv.QUOTE_NONE)
    total_row_count = meta_csv.shape[0]
    valid_rows = meta_csv
    valid_rows = valid_rows[valid_rows['has_grobid_text'] == True]
    return total_row_count, valid_rows


def filter_metadata(raw_meta_path, filtered_meta_path, filter_nan_cols=None, filter_method=None):
    """
    Look at the paper meta data and print out the metadata for the papers
    from different hosting services: arxiv, pubmed, acl-anthalogy etc.
    :param raw_meta_path:
    :param filtered_meta_path:
    :param filter_nan_cols: list(column names); column names based on which to exclude row
        if it contains a nan value.
    :param filter_method: string; {'Computer science', 'Materials science', 'full text'}
    :return:
    """
    if filter_method == 'full text':
        filt_function = filter_for_fulltext
    else:
        raise ValueError('Dont know what filter function to pick.')
    raw_metadata_files = os.listdir(raw_meta_path)
    output_tsv = []
    total_rows = 0
    print('Filtering metadata in: {:s}'.format(raw_meta_path))
    print('Filtering by columns: {:}'.format(filter_nan_cols))
    di = du.DirIterator(root_path=raw_meta_path, yield_list=raw_metadata_files,
                        args=(filter_nan_cols,))
    # Start a pool of worker processes.
    process_pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=10000)
    start = time.time()
    for total_row_count, valid_rows in process_pool.imap_unordered(filt_function, di,
                                                                   chunksize=mp.cpu_count()):
        total_rows += total_row_count
        print('meta_csv: {:}; valid: {:}'.format(total_rows, valid_rows.shape))
        if valid_rows.shape[0] > 0:
            output_tsv.append(valid_rows)
    # Close the pool.
    process_pool.close()
    process_pool.join()
    output_tsv = pd.concat(output_tsv)
    print('Total rows: {:d}; filtered rows: {:}'.format(total_rows, output_tsv.shape))
    if filter_method == 'Computer science' and filter_nan_cols:
        filt_file = os.path.join(filtered_meta_path, 'metadata-{:s}-cs.tsv'.format('-'.join(filter_nan_cols)))
    elif filter_method == 'Materials science':
        filt_file = os.path.join(filtered_meta_path, 'metadata-gorcmatsci.tsv')
    elif filter_method == 'full text':
        filt_file = os.path.join(filtered_meta_path, 'metadata-gorcfulltext.tsv')
    else:
        filt_file = os.path.join(filtered_meta_path, 'metadata-{:s}.tsv'.format('-'.join(filter_nan_cols)))
    output_tsv.to_csv(filt_file, sep='\t')
    print('Wrote: {:s}'.format(filt_file))
    print('Took: {:.4f}s'.format(time.time()-start))
    
    
def write_batch_papers(args):
    """
    Given a batch file, read the papers from it mentioned in the metadatadf
    and write it to disk as a jsonl file.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param pids: pids of the papers we want from the current batch file.
    :return: wrote_count: int; how many jsonl rows were written to the batch output.
    """
    jsonl_fname, pids, filtered_data_path = args
    batch_num = int(os.path.basename(jsonl_fname)[:-9])
    if len(pids) > 0:
        data_file = gzip.open(jsonl_fname)
        out_file = codecs.open(os.path.join(filtered_data_path, '{:d}.jsonl'.format(batch_num)), 'w', 'utf-8')
        for line in data_file:
            data_json = json.loads(line.strip())
            if int(data_json['paper_id']) in pids:
                out_file.write(json.dumps(data_json)+'\n')
        out_file.close()
        return len(pids)
    else:
        return 0
        
    
def gather_papers(meta_fname, raw_data_path):
    """
    Read metadata for (filtered) files and gather the filtered files from the full
    collection.
    :return:
    """
    # Construct output dir path by removing "meta" and ".tsv" from end.
    filtered_data_path = os.path.join(os.path.dirname(meta_fname), os.path.basename(meta_fname)[4:-4])
    du.create_dir(filtered_data_path)
    metadata_df = pd.read_csv(meta_fname, delimiter='\t', error_bad_lines=False,
                              engine='python', quoting=csv.QUOTE_NONE)
    # Get the papers with full text + section labels; include grobid parses also.
    # metadata_df = metadata_df[metadata_df['has_latex'] == True]
    unique_batch_fnames = ['{:d}.jsonl.gz'.format(bid) for bid in metadata_df['batch_num'].unique()]
    di = du.DirMetaIterator(root_path=raw_data_path, yield_list=unique_batch_fnames, metadata_df=metadata_df,
                            args=(filtered_data_path,))
    # Start a pool of worker processes.
    process_pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=10000)
    start = time.time()
    gathered_total = 0
    print('Gathering data from: {:s}; Shape: {:}'.format(meta_fname, metadata_df.shape))
    # Open it in the child processes cause the meta file can be too big to pass
    # with pickle files.
    for wrote_count in process_pool.imap_unordered(write_batch_papers, di,
                                                   chunksize=mp.cpu_count()):
        gathered_total += wrote_count
        print('Wrote rows: {:d}'.format(wrote_count))
    print('Wrote papers to: {:s}'.format(filtered_data_path))
    print('Wrote papers: {:d}'.format(gathered_total))
    print('Took: {:.4f}s'.format(time.time()-start))
    # Close the pool.
    process_pool.close()
    process_pool.join()


def exclude_abstract(abstract_sents):
    """
    Given a json string check if it has everything an example should and return filtered dict.
    :param abstract_sents: list(string)
    :return: bool;
        True if the abstract looks noisey (too many sents or too many tokens in a sentence)
        False if things look fine.
    """
    abs_sent_count = len(abstract_sents)
    if abs_sent_count < pps.MIN_ABS_LEN or abs_sent_count > pps.MAX_ABS_LEN:
        return True
    # Keep count of how many sentences in an abstract and how many tokens in a sentence.
    all_small_sents = True
    for sent in abstract_sents:
        num_toks = len(sent.split())
        if num_toks > pps.MIN_NUM_TOKS:
            all_small_sents = False
        if num_toks > pps.MAX_NUM_TOKS:
            return True
    # If all the sentences are smaller than a threshold then exclude the abstract.
    if all_small_sents:
        return True
    return False
    

def write_batch_absmeta(args):
    """
    Given a batch file, read the papers from it mentioned in the pids,
    filter out obviously noisey papers and write out the title and abstract
    and limited metadata to disk.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param to_write_pids: pids of the papers we want from the current batch file.
    :return:
        to_write_pids: list(int); to write pids.
        pids_written: list(string); actually written pids.
    """
    jsonl_fname, to_write_pids, filtered_data_path = args
    batch_num = int(os.path.basename(jsonl_fname)[:-9])
    pids_written = set()
    if len(to_write_pids) < 0:
        return 0, pids_written
    data_file = gzip.open(jsonl_fname)
    out_file = codecs.open(os.path.join(filtered_data_path, '{:d}.jsonl'.format(batch_num)), 'w', 'utf-8')
    for line in data_file:
        data_json = json.loads(line.strip())
        # The pids comes from metadata which saves it as an integer.
        if int(data_json['paper_id']) not in to_write_pids:
            continue
        # Get title and abstract.
        title_sent = data_json['metadata'].pop('title', None)
        # Assuming this is present in the metadata; Suspect this is only if its gold and provided.
        abstract_sents = []
        try:
            abstrast_str = data_json['metadata'].pop('abstract', None)
            abstract_sents = scispacy_model(abstrast_str,
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            abstract_sents = [sent.text for sent in abstract_sents.sents]
        # Sometimes abstract is missing (is None) in the metadata.
        except TypeError:
            try:
                for abs_par_dict in data_json['grobid_parse']['abstract']:
                    par_sents = scispacy_model(abs_par_dict['text'],
                                               disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                        'lemmatizer', 'parser', 'ner'])
                    par_sents = [sent.text for sent in par_sents.sents]
                    abstract_sents.extend(par_sents)
            # Sometimes, abstract is altogether missing.
            except TypeError:
                pass
        if title_sent == None or abstract_sents == []:
            continue
        # Filter out abstrasts which are noisey.
        if exclude_abstract(abstract_sents):
            continue
        pids_written.add(data_json['paper_id'])
        out_dict = {
            'paper_id': data_json['paper_id'],
            'metadata': data_json['metadata'],
            'title': title_sent,
            'abstract': abstract_sents
        }
        out_file.write(json.dumps(out_dict)+'\n')
        # if len(pids_written) > 20:
        #     break
    out_file.close()
    return to_write_pids, pids_written
    
    
def cocit_corpus_to_jsonl(meta_path, batch_data_path, root_path, out_path, area):
    """
    Given the co-citation information (which sets of papers are co-cited), write out a jsonl
    file with the abstracts and the metadata based on which training data for model will
    be formed (this will still need subsampling and additional cocitation stats based filtering)
    Also filter out data which is obviously noisey in the process.
    In multiprocessing each thread will write one jsonl. In the end, using bash to merge
    all the jsonl files into one jsonl file.
    :param meta_path: strong; directory with pid2citcount files.
    :param batch_data_path: string; directoy with batched jsonl files.
    :param root_path: string; top level directory with pid2batch file.
    :param out_path: string; directory to write batch jsonl files to. Also where filtered citations
        get written.
    :param area: string; {'compsci', 'biomed'}
    :return: writes to disk.
    """
    batch_out_path = os.path.join(out_path, 'batch_data')
    du.create_dir(batch_out_path)
    with codecs.open(os.path.join(root_path, 'pid2batch.json'), 'r', 'utf-8') as fp:
        pid2batch = json.load(fp)
        print('Read: {:s}'.format(fp.name))
        print(f'pid2batch: {len(pid2batch)}')
    with codecs.open(os.path.join(meta_path, f'cocitpids2contexts-{area}.pickle'), 'rb') as fp:
        cocitpids2contexts = pickle.load(fp)
        print('Read: {:s}'.format(fp.name))
    # Get all co-cited papers.
    co_cited_pids = set()
    for cocited_tuple in cocitpids2contexts.keys():
        co_cited_pids.update(cocited_tuple)
    # Get the batch numbers for the pids.
    batch2pids = collections.defaultdict(list)
    missing = 0
    for pid in co_cited_pids:
        try:
            batch_num = pid2batch[pid]
            batch2pids[batch_num].append(pid)
        except KeyError:
            missing += 1
            continue
    batch2pids = dict(batch2pids)
    print(f'Total unique co-cited docs: {len(co_cited_pids)}; Missing in map: {missing}')
    print(f'Number of batches: {len(batch2pids)}')
    del pid2batch
    unique_batch_fnames = ['{:d}.jsonl.gz'.format(bid) for bid in batch2pids.keys()]
    di = du.DirMetaIterator(root_path=batch_data_path, yield_list=unique_batch_fnames, metadata_df=batch2pids,
                            args=(batch_out_path,))
    # Start a pool of worker processes.
    process_pool = mp.Pool(processes=mp.cpu_count()//2, maxtasksperchild=10000)
    start = time.time()
    processed_total = 0
    written_total = 0
    all_written_pids = set()
    for batch_to_writepids, batch_written_pids in process_pool.imap_unordered(write_batch_absmeta, di,
                                                                              chunksize=mp.cpu_count()//2):
        all_written_pids.update(batch_written_pids)
        processed_total += len(batch_to_writepids)
        written_total += len(batch_written_pids)
        print('Processed: {:d} Written: {:d}'.format(len(batch_to_writepids), len(batch_written_pids)))
    # Close the pool.
    process_pool.close()
    process_pool.join()
    
    # Exclude pids which were excluded.
    cocitedpids2contexts_filt = {}
    for cocit_pids, citcontexts in cocitpids2contexts.items():
        filt_cocit_pids = []
        for ccpid in cocit_pids:
            if ccpid not in all_written_pids:
                continue
            else:
                filt_cocit_pids.append(ccpid)
        if len(filt_cocit_pids) > 1:
            cocitedpids2contexts_filt[tuple(filt_cocit_pids)] = citcontexts
    
    # Write out filtered co-citations and their stats.
    with codecs.open(os.path.join(out_path, f'cocitpids2contexts-{area}-absfilt.pickle'), 'wb') as fp:
        pickle.dump(cocitedpids2contexts_filt, fp)
        print(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(out_path, f'cocitpids2contexts-{area}-absfilt.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(cocitedpids2contexts_filt.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        print(f'Wrote: {fp.name}')
    num_cocited_pids = []
    num_citcons = []
    for cocitpids, citcontexts in cocitedpids2contexts_filt.items():
        num_cocited_pids.append(len(cocitpids))
        num_citcons.append(len(citcontexts))
    all_summ = pd.DataFrame(num_cocited_pids).describe()
    print('Papers co-cited together:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_cocited_pids)))
    all_summ = pd.DataFrame(num_citcons).describe()
    print('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))
    
    print('Unfiltered: {:d} Filtered written papers: {:d}'.format(processed_total, written_total))
    print('Unfiltered cocited sets: {:d}; Filtered cocited sets: {:d}'.
          format(len(cocitpids2contexts), len(cocitedpids2contexts_filt)))
    print('Took: {:.4f}s'.format(time.time() - start))
    

def gather_paper_batches(in_path, out_path):
    """
    For the entire GORC corpus build a map of batch to paper id.
    :return:
    """
    batch_fnames = os.listdir(in_path)
    
    batch2pid = {}
    total_papers = 0
    pid2batch = []
    start = time.time()
    for bi, bfname in enumerate(batch_fnames):
        meta_csv = pd.read_csv(os.path.join(in_path, bfname), delimiter='\t', error_bad_lines=False,
                               engine='python', quoting=csv.QUOTE_NONE)
        pids = meta_csv['pid'].tolist()
        batch_num = int(bfname[:-4])
        batch2pid[batch_num] = pids
        total_papers += len(pids)
        pid2batch.extend([(pid, batch_num) for pid in pids])
        if bi % 100 == 0:
            print('batch: {:d}; total_papers: {:d}'.format(bi, total_papers))
    print('Total papers: {:d}'.format(total_papers))
    with codecs.open(os.path.join(out_path, 'pid2batch.json'), 'w', 'utf-8') as fp:
        pid2batch = dict(pid2batch)
        json.dump(pid2batch, fp)
        print('pid2batch: {:d}'.format(len(pid2batch)))
        print('Wrote: {:s}'.format(fp.name))
    with codecs.open(os.path.join(out_path, 'batch2pids.json'), 'w', 'utf-8') as fp:
        json.dump(batch2pid, fp)
        print('batch2pid: {:d}'.format(len(batch2pid)))
        print('Wrote: {:s}'.format(fp.name))
    print('Took: {:.4f}s'.format(time.time() - start))
    
    
def get_citation_count_large(query_meta_row, data_json):
    """
    Given the metadata row for the paper making the citations and the
    full text json data, return the outgoing citation contexts counts.
    :param query_meta_row: dict(); Generated from a pd.Series.
    :param data_json: dict(); full paper dict from batch jsonl.
    :return:
    """
    # Sometimes the citations are NaN
    try:
        # Use the grobid ones because thats used to parse the text.
        outbound_cits = ast.literal_eval(query_meta_row['grobid_bib_links'])
    except ValueError:
        return {}, {}
    # Sometimes its an empty list.
    if not outbound_cits:
        return {}, {}
    # Find the citation contexts in the passed json.
    parsed_paper = data_json['grobid_parse']
    # Get the mapping from bibid to the paper id in the dataset.
    linked_bibid2pid = {}
    for bibid, bibmetadata in parsed_paper['bib_entries'].items():
        if bibmetadata['links']:
            linked_bibid2pid[bibid] = bibmetadata['links']
    # Go over the citations and count up how often they occur in the text.
    # Only the linked citations will be counted up I think.
    pid2citcount = collections.defaultdict(int)
    # Each list element here will be (par_number, sentence_number, sentence_context)
    pid2citcontext = collections.defaultdict(list)
    for par_i, par_dict in enumerate(parsed_paper['body_text']):
        par_text = par_dict['text']
        par_sentences = scispacy_model(par_text,
                                       disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                'lemmatizer', 'parser', 'ner'])
        par_sentences = [sent.text for sent in par_sentences.sents]
        for cit_span in par_dict['cite_spans']:
            # Check for the refid being in the linked bib2pids.
            if cit_span['ref_id'] and cit_span['ref_id'] in linked_bibid2pid:
                cit_span_text = par_text[cit_span['start']:cit_span['end']]
                pid = linked_bibid2pid[cit_span['ref_id']]
                pid2citcount[pid] += 1
                for sent_i, sent in enumerate(par_sentences):
                    if cit_span_text in sent:
                        context_tuple = (par_i, sent_i, sent)
                        pid2citcontext[pid].append(context_tuple)
    return dict(pid2citcount), dict(pid2citcontext)


def write_batch_citation_contexts(args):
    """
    Given a batch file, read the papers from it mentioned in the metadatadf
    and write sentence contexts of outgoing citations.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param pids: pids of the papers we want from the current batch file.
    :return: wrote_count: int; how many jsonl rows were written to the batch output.
    """
    jsonl_fname, pids, batch_metadat_df, filtered_data_path = args
    batch_num = int(os.path.basename(jsonl_fname)[:-6])  # Its 'batch_num.jsonl'
    if len(pids) > 0:
        data_file = codecs.open(jsonl_fname, 'r', 'utf-8')
        citcontextf = codecs.open(os.path.join(filtered_data_path, 'pid2citcontext-{:d}.jsonl'.
                                               format(batch_num)), 'w', 'utf-8')
        citcountf = codecs.open(os.path.join(filtered_data_path, 'pid2citcount-{:d}.jsonl'.
                                             format(batch_num)), 'w', 'utf-8')
        pid2jsonlidx = {}
        total_papers = 0
        valid_papers = 0
        for line in data_file:
            data_json = json.loads(line.strip())
            if int(data_json['paper_id']) in pids:
                row = batch_metadat_df[batch_metadat_df['pid'] == int(data_json['paper_id'])]
                assert(row.empty == False)
                row = row.to_dict('records')
                assert(len(row) == 1)
                row = row[0]
                total_papers += 1
                citation_counts, citation_contexts = get_citation_count_large(
                    query_meta_row=row, data_json=data_json)
                if len(citation_counts) == 0:
                    continue
                pid2jsonlidx[row['pid']] = valid_papers
                valid_papers += 1
                citcontextf.write(json.dumps({row['pid']: citation_contexts})+'\n')
                citcountf.write(json.dumps({row['pid']: citation_counts})+'\n')
            # if valid_papers > 20:
            #     break
        with codecs.open(os.path.join(filtered_data_path, 'pid2jsonlidx-{:d}.json'.format(batch_num)),
                         'w', 'utf-8') as fp:
            json.dump(pid2jsonlidx, fp)
        citcontextf.close()
        citcountf.close()
        return total_papers, valid_papers
    else:
        return 0, 0


def gather_from_citationnw_large(filt_data_path, meta_fname):
    """
    Open up a metadata file of a host-service-filtered subset of the gorc dataset and
    check if the cited file is part of the gorc data and count the number of times
    a cited paper is cited in the query paper and the citation contexts it is in and
    write out these counts for a set of query papers.
    Write out citation contexts and counts as per line jsons for a huge dataset
    and per batch which can them be merged with bash and python scripts (for the pid2idx file).
    :param filt_data_path:
    :param meta_fname: metadata file to gather cited papers for.
    :return:
    """
    query_meta = pd.read_csv(meta_fname, delimiter='\t', error_bad_lines=False,
                             engine='python', quoting=csv.QUOTE_NONE)
    unique_batch_fnames = ['{:d}.jsonl'.format(bid) for bid in query_meta['batch_num'].unique()]
    di = du.DirMetaIterator(root_path=filt_data_path, yield_list=unique_batch_fnames, metadata_df=query_meta,
                            args=(filt_data_path,), yield_meta=True)
    process_pool = mp.Pool(processes=mp.cpu_count()//2, maxtasksperchild=10000)
    start = time.time()
    total_papers = 0
    valid_papers = 0
    for batch_processed_papers, batch_valid_papers in process_pool.imap_unordered(
            write_batch_citation_contexts, di, chunksize=mp.cpu_count()//2):
        total_papers += batch_processed_papers
        valid_papers += batch_valid_papers
        print('Wrote rows: {:d}'.format(valid_papers))
    # Close the pool.
    process_pool.close()
    process_pool.join()
    print('Examined papers: {:d}; Valid query papers: {:d}'.format(total_papers, valid_papers))
    print('Took: {:.4f}s'.format(time.time() - start))


def get_filtbatch_citation_contexts(args):
    """
    Given a batch file, read the citation context jsonls for the pids in the filtered
    batch and return those cit contexts.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param pids: pids of the papers we want from the current batch file.
    :return:
        writes outgoing cits for the area out to disk in a jsonl
            - can be merged with bash after.
        valid_citing_papers; number of citing papers written out.
        outgoing_cits: set(string) pids for papers which are cited.
    """
    citcontext_jsonl_fname, filt_pids, filtered_data_path, area = args
    batch_num = int(os.path.basename(citcontext_jsonl_fname)[15:-6])  # Its 'pid2citcontext-{:d}.jsonl'
    if len(filt_pids) > 0:
        data_file = codecs.open(citcontext_jsonl_fname, 'r', 'utf-8')
        citcontextf = codecs.open(os.path.join(filtered_data_path, f'pid2citcontext-{batch_num}-{area}.jsonl'),
                                  'w', 'utf-8')
        outgoing_cits = set()
        valid_citing_papers = 0
        for line in data_file:
            citcontext_json = json.loads(line.strip())
            assert(len(citcontext_json) == 1)
            citing_pid = list(citcontext_json.keys())[0]
            if int(citing_pid) in filt_pids:
                cited_contexts = list(citcontext_json.values())[0]
                outgoing_cits.update(list(cited_contexts.keys()))
                citcontextf.write(json.dumps(citcontext_json)+'\n')
                valid_citing_papers += 1
            # if valid_citing_papers > 20:
            #     break
        return valid_citing_papers, outgoing_cits
    else:
        return 0, {}
    

def filter_area_citcontexts(filt_data_path, root_path, area):
    """
    - Open metadata file for full-text set of papers and get subset of rows
        which belong to a single area.
    - Send rows which are from the same batch to a batch function which returns
        citcontext json lines for the pids which are of the same area.
    - Write out the citcontext lines for the area to one file.
    - Also get a list of all the papers which are outgoing so their metadata
        can be gathered for future use.
    :param filt_data_path: directory with jsonl files with the cotation context.
    :param meta_fname: fulltext metadata file from which to get filtered area metadata.
    :param root_path: directory where outgoing cit pids are written.
    :param area: {'biomed', 'compsci'}
    :return:
    """
    # The area metadata files are written to disk a-priori from the ipython shell.
    meta_fname = os.path.join(root_path, f'metadata-gorcfulltext-{area}.tsv')
    area_meta = pd.read_csv(meta_fname, delimiter='\t', error_bad_lines=False,
                            engine='python', quoting=csv.QUOTE_NONE)
    unique_batch_fnames = [f'pid2citcontext-{bid}.jsonl' for bid in area_meta['batch_num'].unique()]
    di = du.DirMetaIterator(root_path=filt_data_path, yield_list=unique_batch_fnames, metadata_df=area_meta,
                            args=(filt_data_path, area), yield_meta=False)
    process_pool = mp.Pool(processes=mp.cpu_count()//2, maxtasksperchild=10000)
    start = time.time()
    valid_citing_papers = 0
    outgoing_cits = set()
    for batch_citing_paper_count, batch_outgoing_cits in process_pool.imap_unordered(
            get_filtbatch_citation_contexts, di, chunksize=mp.cpu_count()//2):
        valid_citing_papers += batch_citing_paper_count
        print('Wrote rows: {:d}'.format(batch_citing_paper_count))
        outgoing_cits.update(batch_outgoing_cits)
    with open(os.path.join(root_path, f'outgoing-citpids-{area}.pickle'), 'wb') as fp:
        pickle.dump(outgoing_cits, fp)
        print(f'Wrote: {fp.name}')
    # Close the pool.
    process_pool.close()
    process_pool.join()
    print(f'Area metadata: {area_meta.shape}')
    print(f'Valid query papers: {valid_citing_papers}')
    print(f'Total unique outgoing citations: {len(outgoing_cits)}')
    print('Took: {:.4f}s'.format(time.time() - start))
    
    
def gather_cocitations(root_path, area):
    """
    - Read in citation contexts.
    - Go over the citation contexts and group them into co-citations.
    - Compute stats.
    - Save co-citations to disk.
    """
    citation_contexts = codecs.open(os.path.join(root_path, f'pid2citcontext-{area}.jsonl'), 'r', 'utf-8')
    
    all_cocitedpids2contexts = collections.defaultdict(list)
    singlecited2contexts = collections.defaultdict(list)
    examined_papers = 0
    for citcon_line in citation_contexts:
        if examined_papers % 1000 == 0:
            print(f'Examined papers: {examined_papers}')
        citcond = json.loads(citcon_line.strip())
        citing_pid, cited2contexts = list(citcond.keys())[0], list(citcond.values())[0]
        paper_co_citations = collections.defaultdict(list)
        # Go over all the cited papers and get the co-citations by sentence position.
        for cited_pid, context_tuples in cited2contexts.items():
            # Cited papers can have multiple instances in the citing paper.
            for ct in context_tuples:  # ct is (par_i, sent_i, sent)
                par_i, sent_i, con_sent = ct[0], ct[1], ct[2]
                # Papers in the same sentence are co-cited.
                paper_co_citations[(par_i, sent_i)].append((cited_pid, con_sent))
        # Gather the co-cited papers by pid.
        paper_cocitpids2contexts = collections.defaultdict(list)
        for co_cited_tuple in paper_co_citations.values():
            # There has to be one element atleast and all of the sents will be the same.
            cit_sent = co_cited_tuple[0][1]
            # There can be repeated citations of the same thing in the same sentence
            # or somehow multiple instances of the same pid occur in the parsed spans.
            co_cited_pids = list(set([t[0] for t in co_cited_tuple]))
            co_cited_pids.sort()
            # The same co-cited set of pids in a paper may have mulitiple diff
            # cit contexts. Gather those here.
            paper_cocitpids2contexts[tuple(co_cited_pids)].append((citing_pid, cit_sent))
        # Merge the co-citations across the corpus.
        for cocitpids, citcontexts in paper_cocitpids2contexts.items():
            # Use this if writing to a json file instead of pickle.
            # cocitpids_key = '-'.join(list(cocitpids))
            if len(cocitpids) == 1:
                singlecited2contexts[cocitpids].extend(citcontexts)
            else:
                all_cocitedpids2contexts[cocitpids].extend(citcontexts)
        examined_papers += 1
        # if examined_papers > 50000:
        #     break
    # Write out single citations and their stats.
    with codecs.open(os.path.join(root_path, f'singlecitpids2contexts-{area}.pickle'), 'wb') as fp:
        pickle.dump(singlecited2contexts, fp)
        print(f'Wrote: {fp.name}')
    num_sincited_pids = []
    num_sincitcons = []
    for cocitpids, citcontexts in singlecited2contexts.items():
        num_sincited_pids.append(len(cocitpids))
        num_sincitcons.append(len(citcontexts))
    all_summ = pd.DataFrame(num_sincitcons).describe()
    print('Single papers cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_sincitcons)))

    # Write out co-citations and their stats.
    with codecs.open(os.path.join(root_path, f'cocitpids2contexts-{area}.pickle'), 'wb') as fp:
        pickle.dump(all_cocitedpids2contexts, fp)
        print(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(root_path, f'cocitpids2contexts-{area}.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(all_cocitedpids2contexts.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        print(f'Wrote: {fp.name}')
    num_cocited_pids = []
    num_citcons = []
    for cocitpids, citcontexts in all_cocitedpids2contexts.items():
        num_cocited_pids.append(len(cocitpids))
        num_citcons.append(len(citcontexts))
    all_summ = pd.DataFrame(num_cocited_pids).describe()
    print('Papers co-cited together:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_cocited_pids)))
    all_summ = pd.DataFrame(num_citcons).describe()
    print('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))
    

def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')
    
    # Filter the metadata to group them by hosting service.
    filter_hostserv = subparsers.add_parser('filter_by_hostserv')
    filter_hostserv.add_argument('-i', '--raw_meta_path', required=True,
                                 help='Directory with batchwise metadata.')
    filter_hostserv.add_argument('-o', '--filt_meta_path', required=True,
                                 help='Directory where filtered metadata files should get written.')
    filter_hostserv.add_argument('-d', '--dataset', required=True,
                                 choices=['gorcfulltext'],
                                 help='Directory where split files should get written.')
    # Gather filtered papers.
    gather_hostserv = subparsers.add_parser('gather_by_hostserv')
    gather_hostserv.add_argument('-i', '--in_meta_path', required=True,
                                 help='Directory with a filtered metadata tsv file.')
    gather_hostserv.add_argument('-o', '--raw_data_path', required=True,
                                 help='Directory where batches of raw data.')
    gather_hostserv.add_argument('-d', '--dataset', required=True,
                                 choices=['gorcfulltext'],
                                 help='Directory where split files should get written.')
    # Gather pids and batches.
    batch_pids = subparsers.add_parser('get_batch_pids')
    batch_pids.add_argument('-i', '--in_path', required=True,
                            help='Directory with a batched tsv files.')
    batch_pids.add_argument('-o', '--out_path', required=True,
                            help='Directory to write batch to pid maps.')
    # Gather pids and batches.
    gather_citnw = subparsers.add_parser('gather_from_citationnw')
    gather_citnw.add_argument('-r', '--root_path', required=True,
                              help='Directory metadata, paper data and where outputs should be written.')
    gather_citnw.add_argument('-d', '--dataset', required=True,
                              choices=['gorcfulltext'])
    # Filter co-citation contexts.
    filter_citcon_area = subparsers.add_parser('filter_area_citcontexts')
    filter_citcon_area.add_argument('--root_path', required=True,
                                    help='Directory with metadata, paper data and where '
                                         'outputs should be written.')
    filter_citcon_area.add_argument('--area', required=True,
                                    choices=['compsci', 'biomed', 'matsci'])
    # Gather co-citation contexts.
    gather_cocit_cons = subparsers.add_parser('gather_area_cocits')
    gather_cocit_cons.add_argument('--root_path', required=True,
                                   help='Directory with metadata, paper data and where '
                                        'outputs should be written.')
    gather_cocit_cons.add_argument('--area', required=True,
                                   choices=['compsci', 'biomed', 'matsci'])
    gather_cocitjsonl = subparsers.add_parser('gather_filtcocit_corpus')
    gather_cocitjsonl.add_argument('--root_path', required=True,
                                   help='Directory with pid2batch.')
    gather_cocitjsonl.add_argument('--in_meta_path', required=True,
                                   help='Directory with a filtered metadata tsv file.')
    gather_cocitjsonl.add_argument('--raw_data_path', required=True,
                                   help='Directory where batches of raw data.')
    gather_cocitjsonl.add_argument('--out_path', required=True,
                                   help='Directory where batches of title/abstract jsonl files '
                                        'and filtered citation map should be written.')
    gather_cocitjsonl.add_argument('--dataset', required=True,
                                   choices=['s2orcbiomed', 's2orccompsci', 's2orcmatsci'],
                                   help='Dataset for which outputs should be written.')
    cl_args = parser.parse_args()
    
    if cl_args.subcommand == 'filter_by_hostserv':
        if cl_args.dataset == 'gorcfulltext':
            filter_metadata(raw_meta_path=cl_args.raw_meta_path,
                            filtered_meta_path=cl_args.filt_meta_path,
                            filter_method='full text')
    elif cl_args.subcommand == 'gather_by_hostserv':
        if cl_args.dataset in {'gorcfulltext'}:
            meta_fname = os.path.join(cl_args.in_meta_path, 'metadata-{:s}.tsv'.format(cl_args.dataset))
        gather_papers(meta_fname=meta_fname, raw_data_path=cl_args.raw_data_path)
    elif cl_args.subcommand == 'get_batch_pids':
        # Run once for the entire gorc corpus, no need to re-run over and over.
        gather_paper_batches(in_path=cl_args.in_path, out_path=cl_args.out_path)
    elif cl_args.subcommand == 'gather_from_citationnw':
        filt_root_path = os.path.join(cl_args.root_path, 'hostservice_filt')
        if cl_args.dataset == 'gorcfulltext':
            meta_fname = os.path.join(filt_root_path, 'metadata-gorcfulltext.tsv')
            batch_data_path = os.path.join(filt_root_path, 'data-gorcfulltext')
            gather_from_citationnw_large(filt_data_path=batch_data_path, meta_fname=meta_fname)
    elif cl_args.subcommand == 'filter_area_citcontexts':
        filt_root_path = os.path.join(cl_args.root_path, 'hostservice_filt')
        batch_data_path = os.path.join(filt_root_path, 'data-gorcfulltext')
        filter_area_citcontexts(filt_data_path=batch_data_path, area=cl_args.area,
                                root_path=filt_root_path)
    elif cl_args.subcommand == 'gather_area_cocits':
        filt_root_path = os.path.join(cl_args.root_path, 'hostservice_filt')
        gather_cocitations(root_path=filt_root_path, area=cl_args.area)
    elif cl_args.subcommand == 'gather_filtcocit_corpus':
        if cl_args.dataset == 's2orccompsci':
            area = 'compsci'
        elif cl_args.dataset == 's2orcbiomed':
            area = 'biomed'
        elif cl_args.dataset == 's2orcmatsci':
            area = 'matsci'
        cocit_corpus_to_jsonl(meta_path=cl_args.in_meta_path, batch_data_path=cl_args.raw_data_path,
                              out_path=cl_args.out_path, area=area, root_path=cl_args.root_path)


if __name__ == '__main__':
    main()
