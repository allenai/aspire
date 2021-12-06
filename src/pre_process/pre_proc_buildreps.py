"""
Build abstract or sentence embeddings from own trained models or a pretrained model
and save to disk to use for ranking.
"""
import os
import sys
import logging
import re
import time
import codecs, json
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer, models

from . import data_utils as du
from ..learning.facetid_models import disent_models
from ..learning import batchers

np_random_ng = np.random.default_rng()


class BertMLM:
    def __init__(self, model_name='specter'):
        mapping = {
            'specter': 'allenai/specter',
            # Using roberta here causes the tokenizers below to break cause roberta inputs != bert inputs.
            'supsimcse': 'princeton-nlp/sup-simcse-bert-base-uncased',
            'unsupsimcse': 'princeton-nlp/unsup-simcse-bert-base-uncased'
        }
        full_name = mapping[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(full_name)
        if model_name == 'sentence-transformers/all-mpnet-base-v2':
            self.bert_max_seq_len = 500
        else:
            self.bert_max_seq_len = 500
        self.model = AutoModel.from_pretrained(full_name)
        self.model.config.output_hidden_states = True
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
    
    def _prepare_batch(self, batch):
        """
        Prepare the batch for Bert.
        :param batch: list(string); batch of strings.
        :return:
        """
        # Construct the batch.
        tokenized_batch = []
        batch_seg_ids = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for sent in batch:
            bert_tokenized_text = self.tokenizer.tokenize(sent)
            if len(bert_tokenized_text) > self.bert_max_seq_len:
                bert_tokenized_text = bert_tokenized_text[:self.bert_max_seq_len]
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokenized_text)
            # Append CLS and SEP tokens to the text.
            indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=indexed_tokens)
            if len(indexed_tokens) > max_seq_len:
                max_seq_len = len(indexed_tokens)
            tokenized_batch.append(indexed_tokens)
            batch_seg_ids.append([0] * len(indexed_tokens))
            batch_attn_mask.append([1] * len(indexed_tokens))
        # Pad the batch.
        for ids_sent, seg_ids, attn_mask in \
                zip(tokenized_batch, batch_seg_ids, batch_attn_mask):
            pad_len = max_seq_len - len(ids_sent)
            seq_lens.append(len(ids_sent))
            ids_sent.extend([self.tokenizer.pad_token_id] * pad_len)
            seg_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            attn_mask.extend([self.tokenizer.pad_token_id] * pad_len)
        return torch.tensor(tokenized_batch), torch.tensor(batch_seg_ids), \
               torch.tensor(batch_attn_mask), torch.FloatTensor(seq_lens)
        
    def predict(self, batch):
        """
        :param batch:
        :return:
        """
        tokid_tt, seg_tt, attnmask_tt, seq_lens_tt = self._prepare_batch(batch)
        if torch.cuda.is_available():
            tokid_tt = tokid_tt.cuda()
            seg_tt = seg_tt.cuda()
            attnmask_tt = attnmask_tt.cuda()
            seq_lens_tt = seq_lens_tt.cuda()
        
        with torch.no_grad():
            model_out = self.model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
            # top_l is [bs x max_seq_len x bert_encoding_dim]
            top_l = model_out.last_hidden_state
            batch_reps_cls = top_l[:, 0, :]
            batch_reps_av = torch.sum(top_l[:, 1:-1, :], dim=1)
            batch_reps_av = batch_reps_av / seq_lens_tt.unsqueeze(dim=1)
        if torch.cuda.is_available():
            batch_reps_av = batch_reps_av.cpu().data.numpy()
            batch_reps_cls = batch_reps_cls.cpu().data.numpy()
        return batch_reps_av, batch_reps_cls


class SimCSE(BertMLM):
    def predict(self, batch):
        """
        :param batch:
        :return:
        """
        tokid_tt, seg_tt, attnmask_tt, seq_lens_tt = self._prepare_batch(batch)
        if torch.cuda.is_available():
            tokid_tt = tokid_tt.cuda()
            seg_tt = seg_tt.cuda()
            attnmask_tt = attnmask_tt.cuda()
            seq_lens_tt = seq_lens_tt.cuda()
        
        with torch.no_grad():
            model_out = self.model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
            # top_l is [bs x max_seq_len x bert_encoding_dim]
            top_l = model_out.last_hidden_state
            batch_reps_pooler = model_out.pooler_output
            batch_reps_cls = top_l[:, 0, :]
        if torch.cuda.is_available():
            batch_reps_pooler = batch_reps_pooler.cpu().data.numpy()
            batch_reps_cls = batch_reps_cls.cpu().data.numpy()
        return batch_reps_cls, batch_reps_pooler

    
class TrainedModel:
    """
    Own trained model using which we want to build up document embeddings.
    """
    def __init__(self, model_name, trained_model_path, model_version='cur_best'):
        # Load label maps and configs.
        with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
        # Init model:
        if model_name in {'cospecter'}:
            model = disent_models.MySPECTER(model_hparams=all_hparams)
        else:
            raise ValueError(f'Unknown model: {model_name}')
        model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
        model.load_state_dict(torch.load(model_fname))
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
            logging.info('Running on GPU.')
        model.eval()
        self.model_name = model_name
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(all_hparams['base-pt-layer'])
    
    def predict(self, batch):
        """
        :param batch:
        :return:
        """
        if self.model_name in {'cospecter'}:
            bert_batch, _, _ = batchers.SentTripleBatcher.prepare_bert_sentences(sents=batch, tokenizer=self.tokenizer)
            ret_dict = self.model.encode(batch_dict={'bert_batch': bert_batch})
            return ret_dict, ret_dict['doc_reps']


class SplitStream:
    """
    Given a jsonl file yield text in the corus.
    Returns the title vs the abstract or both based on what is asked.
    """
    
    def __init__(self, in_fname, num_to_read=None, attribute='title',
                 return_pid=False, insert_sep=False):
        """
        :param in_fname: string; input jsonl filename from which to read exampled.
        :param num_to_read: int; number of examples to read.
            None if everything in the file should be read.
        :param attribute: string; which attribute from the input example should be read.
        :param return_pid: bool; Return PID if True else dont.
        :param insert_sep: bool; Insert [SEP] tokens between sentences of the abstract. For use by bert.
        """
        self.in_fname = in_fname
        self.attr_to_read = attribute
        self.num_to_read = num_to_read
        self.return_pid = return_pid
        self.insert_sep = insert_sep
        self.read_count = 0
        
    def __iter__(self):
        # "Rewind" the input file at the start of the loop
        self.in_file = codecs.open(self.in_fname, 'r', 'utf-8')
        return self.next()
    
    def next(self):
        # In each loop iteration return one example.
        for jsonline in self.in_file:
            self.read_count += 1
            if self.num_to_read and self.read_count == self.num_to_read:
                break
            if self.attr_to_read in {'sent'}:
                # If this happens then it is yielding sentences so say next.
                doc = self.get_gorc_sents(jsonline, return_pid=self.return_pid)
                for sent in doc:
                    yield sent
            elif self.attr_to_read in {'abstract'}:
                doc = self.get_gorc(jsonline, attr_to_read=self.attr_to_read,
                                    return_pid=self.return_pid, insert_sep=self.insert_sep)
                # Check to make sure that the text is a non empty string.
                ret_text = doc[1].strip() if self.return_pid else doc.strip()
                if ret_text:
                    yield doc
            elif self.attr_to_read in {'title-abstract'}:
                doc = self.get_gorc_specter(jsonline, return_pid=self.return_pid)
                yield doc
            elif self.attr_to_read in {'title-abstract-dict'}:
                doc = self.get_gorc_absdict(jsonline, return_pid=self.return_pid)
                yield doc
            else:
                raise ValueError('Unknown attribute to read: {:s}'.format(self.attr_to_read))
                
    @staticmethod
    def get_gorc(in_line, attr_to_read, return_pid, insert_sep):
        """
        Read in a gorc doc line of text and return concated sentences.
        Also replace all numbers with <NUM> to match processing in the "Ask the GRU" paper.
        :param in_line: string; json string example.
        :param attr_to_read: string; says what should be read from the json example.
        :param return_pid: bool; Return PID if True else dont.
        :param insert_sep: bool; Insert [SEP] tokens between sentences of the abstract. For use by bert.
        :return:
            if 'abstract': all the sentences of the abstract concated into one string.
            if 'title': the title sentence.
        """
        in_ex = json.loads(in_line.strip())
        pid = in_ex['paper_id']
        if attr_to_read == 'abstract':
            sents = in_ex['abstract']
            if insert_sep:
                ret_text = ' [SEP] '.join(sents)
            else:
                ret_text = ' '.join(sents)
        else:
            raise ValueError('Unknown attribute to read: {:}'.format(attr_to_read))
        # Replace numbers a place holder.
        ret_text = re.sub(r"\d+", "<NUM>", ret_text)
        if return_pid:
            return pid, ret_text
        else:
            return ret_text

    @staticmethod
    def get_gorc_specter(in_line, return_pid):
        """
        Read in a gorc doc line of text and return title and abstract concatenated.
        :param in_line: string; json string example.
        :param attr_to_read: string; says what should be read from the json example.
        :param return_pid: bool; Return PID if True else dont.
        :return:
            if 'abstract': all the sentences of the abstract concated into one string.
        """
        in_ex = json.loads(in_line.strip())
        pid = in_ex['paper_id']
        sents = in_ex['abstract']
        abs_text = ' '.join(sents)
        ret_text = in_ex['title'] + '[SEP]' + abs_text
        if return_pid:
            return pid, ret_text
        else:
            return ret_text

    @staticmethod
    def get_gorc_absdict(in_line, return_pid):
        """
        Read in a gorc doc line of text and return title and abstract in a dict as expected
        by src.learning.batchers.*.prepare_abstracts and others
        :param in_line: string; json string example.
        :param return_pid: bool; Return PID if True else dont.
        :return:
            ret_dict: dict('TITLE': string, 'ABSTRACT': list(string))
        """
        in_ex = json.loads(in_line.strip())
        pid = in_ex['paper_id']
        ret_dict = {'TITLE': in_ex['title'], 'ABSTRACT': in_ex['abstract']}
        if return_pid:
            return pid, ret_dict
        else:
            return ret_dict

    @staticmethod
    def get_gorc_sents(in_line, return_pid):
        """
        Read in a gorc doc line of text and return sentences one at a time.
        :param in_line: string; json string example.
        :param return_pid: bool; Return PID if True else dont.
        :return:
            ret_toks: list(str); tokenized sentence with numbers replaced with num and
                unknown (ie low freq) tokens with unk.
        """
        in_ex = json.loads(in_line.strip())
        pid = in_ex['paper_id']
        sents = in_ex['abstract']
        for i, sent in enumerate(sents):
            if return_pid:
                yield '{:s}-{:d}'.format(pid, i), sent
            else:
                yield sent


def build_sentbert_reps(data_path, run_path, data_to_read, dataset, sb_model_name, trained_model_path=None):
    """
    Build per sentence sentence BERT representations for csfcube.
    :param data_path: string; path from which to read raw abstracts and vocab data.
    :param run_path: string; path to save reps for documents.
    :param data_to_read: string; {'sent'}
    :param dataset: string; {'csfcube', 'relish'}
    :param sb_model_name: string; The original sent bert model trained on NLI alone
        or the one trained on citations+NLI+Paraphrases achieving SOTA
        SciDOCS performance.
    :param trained_model_path: string; directory where torch.save was used to store
        a bert encoder fine tuned on my own data.
    :return:
    """
    if sb_model_name in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B'}:
        normname2model_names = {
            'sbtinybertsota': 'paraphrase-TinyBERT-L6-v2',
            'sbrobertanli': 'nli-roberta-base-v2',
            'sbmpnet1B': 'sentence-transformers/all-mpnet-base-v2'
        }
        pt_model_name = normname2model_names[sb_model_name]
        sentbert_model = SentenceTransformer(pt_model_name)
    # The easy way to get sentence reps from any bert model.
    elif sb_model_name in {'cosentbert', 'ictsentbert'} and trained_model_path:
        word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                                  max_seq_length=512)
        # Loading local model: https://github.com/huggingface/transformers/issues/2422#issuecomment-571496558
        trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
        word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    embedding_dim = 768
    if dataset in {'csfcube'}:
        in_fname = os.path.join(data_path, 'abstracts-{:s}-preds.jsonl'.format(dataset))
    elif dataset in {'relish', 'treccovid', 'gorcmatscicit', 'scidcite',
                     'scidcocite', 'scidcoread', 'scidcoview'}:
        in_fname = os.path.join(data_path, 'abstracts-{:s}.jsonl'.format(dataset))
    out_fname = os.path.join(run_path, '{:s}-{:s}.npy'.format(dataset, data_to_read))
    out_map_fname = os.path.join(run_path, 'pid2idx-{:s}-{:s}.json'.format(dataset, data_to_read))
    abs_sent_stream = SplitStream(in_fname=in_fname, attribute=data_to_read,
                                  return_pid=True)
    sent_docs = list(abs_sent_stream)
    # Docs are returned in the same order.
    pid2idx = {}
    for absi, (pid, abs_sentence) in enumerate(sent_docs):
        pid2idx[pid] = absi
    logging.info('pid2idx: {:}'.format(len(pid2idx)))
    del sent_docs
    # Go over documents and form sb reps for documents.
    abs_sent_stream = list(SplitStream(in_fname=in_fname, attribute=data_to_read))
    start = time.time()
    vectors = sentbert_model.encode(abs_sent_stream)
    logging.info('Forming vectors took: {:.4f}s'.format(time.time() - start))
    logging.info('Shape: {:}'.format(vectors.shape))
    # Save vectors to disk.
    with codecs.open(out_fname, 'wb') as fp:
        np.save(fp, vectors)
        logging.info('Wrote: {:s}'.format(fp.name))
    with codecs.open(out_map_fname, 'w', 'utf-8') as fp:
        json.dump(pid2idx, fp)
        logging.info('Wrote: {:s}'.format(fp.name))
       

def write_wholeabs_reps(data_path, run_path, dataset, model_name, trained_model_path=None):
    """
    Given a corpus: read the abstract sentences and write out the bert representations of
    the abstracts. The entire abstract is passed through bert as one string with [SEP] tokens
    marking off sentences.
    Also (mis)-using this function to get sentence reps for sup/unsupsimcse.
    :param data_path: base directory with abstract jsonl docs.
    :param run_path: directory to which bert reps, and maps of bert reps to strings will be written.
    :param dataset: string; {'relish', 'treccovid', 'csfcube'}
    :param model_name: string; {'specter', 'cospecter'}
    :return: None. Writes to disk.
    """
    sent_enc_dim = 768
    in_fname = os.path.join(data_path, 'abstracts-{:s}.jsonl'.format(dataset))
    cls_out_fname = os.path.join(run_path, '{:s}-abstracts.npy'.format(dataset))
    out_map_fname = os.path.join(run_path, 'pid2idx-{:s}-abstract.json'.format(dataset))
    num_docs = len(list(SplitStream(in_fname=in_fname, attribute='title-abstract', return_pid=True)))
    
    if model_name in {'supsimcse', 'unsupsimcse'}:
        in_fname = os.path.join(data_path, 'abstracts-{:s}.jsonl'.format(dataset))
        # Over write the above values.
        cls_out_fname = os.path.join(run_path, '{:s}-sent.npy'.format(dataset))
        out_map_fname = os.path.join(run_path, 'pid2idx-{:s}-sent.json'.format(dataset))
        num_docs = len(list(SplitStream(in_fname=in_fname, attribute='sent', return_pid=True)))
        doc_stream = SplitStream(in_fname=in_fname, attribute='sent', return_pid=True)
        model = SimCSE(model_name)
        batch_size = 120
    elif model_name in {'specter'}:
        doc_stream = SplitStream(in_fname=in_fname, attribute='title-abstract', return_pid=True)
        model = BertMLM(model_name)
        batch_size = 90
    elif model_name in {'cospecter'}:
        doc_stream = SplitStream(in_fname=in_fname, attribute='title-abstract', return_pid=True)
        model = TrainedModel(model_name=model_name, trained_model_path=trained_model_path)
        batch_size = 32
    start = time.time()
    logging.info('Processing files in: {:s}'.format(in_fname))
    logging.info('Num docs: {:d}'.format(num_docs))
    
    # Write out sentence reps incrementally.
    sent2idx = {}
    doc_reps_cls = np.empty((num_docs, sent_enc_dim))
    logging.info('Allocated space for reps: {:}'.format(doc_reps_cls.shape))
    batch_docs = []
    batch_start_idx = 0
    for doci, (pid, abs_text) in enumerate(doc_stream):
        if doci % 1000 == 0:
            logging.info('Processing document: {:d}/{:d}'.format(doci, num_docs))
        batch_docs.append(abs_text)
        sent2idx[pid] = len(sent2idx)
        if len(batch_docs) == batch_size:
            batch_reps_av, batch_reps_cls = model.predict(batch_docs)
            batch_docs = []
            doc_reps_cls[batch_start_idx:batch_start_idx+batch_size, :] = batch_reps_cls
            batch_start_idx = batch_start_idx+batch_size
    # Handle left over sentences.
    if len(batch_docs) > 0:
        batch_reps_av, batch_reps_cls = model.predict(batch_docs)
        final_bsize = batch_reps_cls.shape[0]
        doc_reps_cls[batch_start_idx:batch_start_idx + final_bsize, :] = batch_reps_cls
    logging.info('Doc reps shape: {:}; Map length: {:d}'.format(doc_reps_cls.shape, len(sent2idx)))
    with codecs.open(out_map_fname, 'w', 'utf-8') as fp:
        json.dump(sent2idx, fp)
        logging.info('Wrote: {:s}'.format(fp.name))
    with codecs.open(cls_out_fname, 'wb') as fp:
        np.save(fp, doc_reps_cls)
        logging.info('Wrote: {:s}'.format(fp.name))
    logging.info('Took: {:.4f}s'.format(time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Get tfidf reps.
    build_vecs_args = subparsers.add_parser('build_reps')
    build_vecs_args.add_argument('--model_name', required=True,
                                 choices=['sbtinybertsota', 'sbrobertanli', 'specter',
                                          'cosentbert', 'ictsentbert', 'cospecter',
                                          'supsimcse', 'unsupsimcse', 'sbmpnet1B'],
                                 help='The name of the model to run.')
    build_vecs_args.add_argument('--dataset', required=True,
                                 choices=['gorcmatscicit', 'csfcube', 'relish', 'treccovid',
                                          'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'],
                                 help='The dataset to train and predict on.')
    build_vecs_args.add_argument('--data_path', required=True,
                                 help='Path to directory with jsonl data.')
    build_vecs_args.add_argument('--run_path', required=True,
                                 help='Path to directory to save all run items to.')
    build_vecs_args.add_argument('--model_path',
                                 help='Path to directory with trained model to use for getting reps.')
    build_vecs_args.add_argument('--run_name',
                                 help='Basename for the trained model directory.')
    build_vecs_args.add_argument('--log_fname',
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
    if cl_args.subcommand == 'build_reps':
        if cl_args.model_name in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B'}:
            build_sentbert_reps(data_path=cl_args.data_path, run_path=cl_args.run_path,
                                data_to_read='sent', dataset=cl_args.dataset,
                                sb_model_name=cl_args.model_name)
        elif cl_args.model_name in {'cosentbert', 'ictsentbert'}:
            # Write reps to a different directory per run.
            run_path = os.path.join(cl_args.run_path, cl_args.run_name)
            du.create_dir(run_path)
            build_sentbert_reps(data_path=cl_args.data_path, run_path=run_path,
                                data_to_read='sent', dataset=cl_args.dataset,
                                sb_model_name=cl_args.model_name,
                                trained_model_path=cl_args.model_path)
        elif cl_args.model_name in {'specter', 'supsimcse', 'unsupsimcse'}:
            write_wholeabs_reps(data_path=cl_args.data_path, run_path=cl_args.run_path,
                                dataset=cl_args.dataset, model_name=cl_args.model_name)
        elif cl_args.model_name in {'cospecter'}:
            # Write reps to a different directory per run.
            run_path = os.path.join(cl_args.run_path, cl_args.run_name)
            du.create_dir(run_path)
            write_wholeabs_reps(data_path=cl_args.data_path, run_path=run_path,
                                dataset=cl_args.dataset,
                                model_name=cl_args.model_name,
                                trained_model_path=cl_args.model_path)


if __name__ == '__main__':
    main()
