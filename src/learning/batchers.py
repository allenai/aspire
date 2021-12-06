"""
Classes to stream int-mapped data from file in batches, pad and sort them (as needed)
and return batch dicts for the models.
"""
import codecs
import sys
import re

import numpy as np
import torch
from transformers import AutoTokenizer

from . import data_utils as du

replace_sep = re.compile(r'\[SEP\]')


class GenericBatcher:
    def __init__(self, num_examples, batch_size):
        """
        Maintain batcher variables, state and such. Any batcher for a specific
        model is a subclass of this and implements specific methods that it
        needs.
        - A batcher needs to know how to read from an int-mapped raw-file.
        - A batcher should yield a dict which you model class knows how to handle.
        :param num_examples: the number of examples in total.
        :param batch_size: the number of examples to have in a batch.
        """
        # Batch sizes book-keeping; the 0 and -1 happen in the case of test time usage.
        if num_examples > 0 and batch_size > -1:
            self.full_len = num_examples
            self.batch_size = batch_size
            if self.full_len > self.batch_size:
                self.num_batches = int(np.ceil(float(self.full_len) / self.batch_size))
            else:
                self.num_batches = 1
    
            # Get batch indices.
            self.batch_start = 0
            self.batch_end = self.batch_size

    def next_batch(self):
        """
        This should yield the dict which your model knows how to make sense of.
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count):
        """
        Implement whatever you need for reading a raw batch of examples.
        Read the next batch from the file.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of examples to read from the file.
        :return:
        """
        raise NotImplementedError


class SentTripleBatcher(GenericBatcher):
    """
    Feeds a model which inputs query, positive. Negatives are in-batch.
    """
    bert_config_str = None
    
    def __init__(self, ex_fnames, num_examples, batch_size):
        """
        Batcher class for the em style trained models.
        This batcher is also used at test time, at this time all the arguments here are
        meaningless. Only the make_batch and ones beneath it will be used.
        :param ex_fnames: dict('pos_ex_fname': str, 'neg_ex_fname': str)
        :param num_examples: int.
        :param batch_size: int.
        :param bert_config: string; BERT config string to initialize tokenizer with.
        :param max_pos_neg: int; maximum number of positive and negative examples per
            query to train with.
        """
        GenericBatcher.__init__(self, num_examples=num_examples,
                                batch_size=batch_size)
        # Call it pos ex fname even so code elsewhere can be re-used.
        if ex_fnames:
            pos_ex_fname = ex_fnames['pos_ex_fname']
            # Access the file with the sentence level examples.
            self.pos_ex_file = codecs.open(pos_ex_fname, 'r', 'utf-8')
        self.pt_lm_tokenizer = AutoTokenizer.from_pretrained(self.bert_config_str)
    
    def next_batch(self):
        """
        Yield the next batch. Based on whether its train_mode or not yield a
        different set of items.
        :return:
            batch_doc_ids: list; with the doc_ids corresponding to the
                    examples in the batch.
            batch_dict: see make_batch.
        """
        for nb in range(self.num_batches):
            # Read the batch of data from the file.
            if self.batch_end < self.full_len:
                cur_batch_size = self.batch_size
            else:
                cur_batch_size = self.full_len - self.batch_start
            batch_query_docids, batch_queries, batch_pos, batch_neg = \
                next(SentTripleBatcher.raw_batch_from_file(self.pos_ex_file, cur_batch_size))
            self.batch_start = self.batch_end
            self.batch_end += self.batch_size
            try:
                if batch_neg and batch_pos:
                    feed = {'query_texts': batch_queries, 'pos_texts': batch_pos, 'neg_texts': batch_neg}
                elif batch_pos:
                    feed = {'query_texts': batch_queries, 'pos_texts': batch_pos}
                else:
                    feed = {'query_texts': batch_queries}
                batch_dict = self.make_batch(raw_feed=feed, pt_lm_tokenizer=self.pt_lm_tokenizer)
            except (IndexError, AssertionError) as error:
                print(batch_query_docids)
                print(batch_queries)
                print(batch_pos)
                sys.exit()
            batch_dict = {
                'batch_rank': batch_dict
            }
            yield batch_query_docids, batch_dict
    
    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count):
        """
        Read the next batch from the file. In reading the examples:
        - For every query only read max_pos_neg positive and negative examples.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of lines to read from the file.
        :return:
            query_abs: list(str); list of query sentences
            pos_abs: list(str); list of positive sentences
            neg_abs: list(str); list of negative sentences
        """
        # Initial values.
        read_ex_count = 0
        # These will be to_read_count long.
        ex_query_docids = []
        query_texts = []
        pos_texts = []
        neg_texts = []
        # Read content from file until the file content is exhausted.
        for ex in du.read_json(ex_file):
            docids = read_ex_count
            ex_query_docids.append(docids)
            query_texts.append(ex['query'])
            # Dont assume even a positive is present -- happens because of
            # SimCSE like pretraining.
            try:
                pos_texts.append(ex['pos_context'])
            except KeyError:
                pass
            # Only dev files have neg examples. Pos used inbatch negs.
            try:
                neg_texts.append(ex['neg_context'])
            except KeyError:
                pass
            read_ex_count += 1
            if read_ex_count == to_read_count:
                yield ex_query_docids, query_texts, pos_texts, neg_texts
                # Once execution is back here empty the lists and reset counters.
                read_ex_count = 0
                ex_query_docids = []
                query_texts = []
                pos_texts = []
                neg_texts = []
    
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        Creates positive and query batches. Only used for training. Test use happens
        with embeddings generated in the pre_proc_buildreps scripts.
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with query sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
                'pos_bert_batch': dict();  The batch which BERT inputs with positive sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        pos_texts = raw_feed['pos_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, _, _ = SentTripleBatcher.prepare_bert_sentences(sents=query_texts, tokenizer=pt_lm_tokenizer)
        pbert_batch, _, _ = SentTripleBatcher.prepare_bert_sentences(sents=pos_texts, tokenizer=pt_lm_tokenizer)

        # Happens with the dev set in models using triple losses and in batch negs.
        if 'neg_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, _, _ = SentTripleBatcher.prepare_bert_sentences(sents=neg_texts, tokenizer=pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch,
                'neg_bert_batch': nbert_batch
            }
        else:
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch
            }
        return batch_dict
    
    @staticmethod
    def prepare_bert_sentences(sents, tokenizer):
        """
        Given a batch of sentences prepare a batch which can be passed through BERT.
        :param sents: list(string)
        :param tokenizer: an instance of the appropriately initialized BERT tokenizer.
        :return:
        """
        max_num_toks = 500
        # Construct the batch.
        tokenized_batch = []
        tokenized_text = []
        batch_seg_ids = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for sent in sents:
            bert_tokenized_text = tokenizer.tokenize(sent)
            bert_tokenized_text = bert_tokenized_text[:max_num_toks]
            tokenized_text.append(bert_tokenized_text)
            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokenized_text)
            # Append CLS and SEP tokens to the text..
            indexed_tokens = tokenizer.build_inputs_with_special_tokens(token_ids_0=indexed_tokens)
            if len(indexed_tokens) > max_seq_len:
                max_seq_len = len(indexed_tokens)
            seq_lens.append(len(indexed_tokens))
            tokenized_batch.append(indexed_tokens)
            batch_seg_ids.append([0] * len(indexed_tokens))
            batch_attn_mask.append([1] * len(indexed_tokens))
        # Pad the batch.
        for ids_sent, seg_ids, attn_mask in zip(tokenized_batch, batch_seg_ids, batch_attn_mask):
            pad_len = max_seq_len - len(ids_sent)
            ids_sent.extend([tokenizer.pad_token_id] * pad_len)
            seg_ids.extend([tokenizer.pad_token_id] * pad_len)
            attn_mask.extend([tokenizer.pad_token_id] * pad_len)
        # The batch which the BERT model will input.
        bert_batch = {
            'tokid_tt': torch.tensor(tokenized_batch),
            'seg_tt': torch.tensor(batch_seg_ids),
            'attnmask_tt': torch.tensor(batch_attn_mask),
            'seq_lens': seq_lens
        }
        return bert_batch, tokenized_text, tokenized_batch


class AbsTripleBatcher(SentTripleBatcher):
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        Creates positive and query batches. Only used for training. Test use happens
        with embeddings generated in the pre_proc_buildreps scripts.
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with query sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
                'pos_bert_batch': dict();  The batch which BERT inputs with positive sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=query_texts, pt_lm_tokenizer=pt_lm_tokenizer)

        # Happens with the dev set in models using triple losses and in batch negs.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=neg_texts, pt_lm_tokenizer=pt_lm_tokenizer)
            pos_texts = raw_feed['pos_texts']
            pbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=pos_texts, pt_lm_tokenizer=pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch,
                'neg_bert_batch': nbert_batch
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=pos_texts, pt_lm_tokenizer=pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch,
            }
        return batch_dict

    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            seqs = [ex_abs['TITLE']]
            seqs.extend([s for s in ex_abs['ABSTRACT']])
            batch_abs_seqs.append(' [SEP] '.join([replace_sep.sub('', s) for s in seqs]))
        bert_batch, tokenized_abs, tokenized_ids = SentTripleBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
        return bert_batch


class AbsSentBatcher(SentTripleBatcher):
    """
    Feeds a model which inputs query, positive and negative abstracts and sentence
    idxs for the abstracts. Negatives only at dev time, else the model uses in-batch
    negatives.
    """
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        - Create [SEP] demarcated abs sents to feed bert.
        - Generate indices to read off sentence reps from bert output.
            (accounting for max sent len and special tokens inserted by BERT tokenizers)
        - Generate indices for facet reps.
        If 'neg_abs' is in the raw_feed this is a call during training else it is
        a call during test and the pos_abs contains candidates to rank wrt the query.
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'query_abs_lens': list(int); Number of sentences in query abs.
                'query_sep_idxs': LongTensor; Indices of the sep tokens to get sent reps,
                    flattened and indices adjusted to index the one dimensional token reps.
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_abs_lens': list(int);
                'pos_sep_idxs': LongTensor; Indices of the sep tokens to get sent reps,
                    flattened and indices adjusted to index the one dimensional token reps.
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_abs_lens': list(int);
                'neg_sep_idxs': LongTensor; Indices of the sep tokens to get sent reps,
                    flattened and indices adjusted to index the one dimensional token reps.
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, qabs_sep_idxs, qabs_len, qabs_flatsep_idxs = AbsSentBatcher.prepare_abstracts(
            query_texts, pt_lm_tokenizer)

        # Happens in the dev set.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, nabs_sep_idxs, nabs_len, nabs_flatsep_idxs = AbsSentBatcher.prepare_abstracts(
                neg_texts, pt_lm_tokenizer)
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_sep_idxs, pabs_len, pabs_flatsep_idxs = AbsSentBatcher.prepare_abstracts(
                pos_texts, pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_sep_idxs': qabs_flatsep_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_sep_idxs': pabs_flatsep_idxs,
                'neg_bert_batch': nbert_batch, 'neg_abs_lens': nabs_len, 'neg_sep_idxs': nabs_flatsep_idxs
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_sep_idxs, pabs_len, pabs_flatsep_idxs = AbsSentBatcher.prepare_abstracts(
                pos_texts, pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_sep_idxs': qabs_flatsep_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_sep_idxs': pabs_flatsep_idxs
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch, 'abs_lens': qabs_len, 'sep_idxs': qabs_flatsep_idxs
            }
        return batch_dict

    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sep_idxs: LongTensor; indices of SEP tokens demarcating sentences in abs adjusted
                to index a one dim token array for the whole batch. Also making sure of pading sents.
            facet_idxs: list(list(list(int))); indices of sentences belonging to the query facet.
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            seqs = [ex_abs['TITLE']]
            seqs.extend([s for s in ex_abs['ABSTRACT']])
            batch_abs_seqs.append(' [SEP] '.join([replace_sep.sub('', s) for s in seqs]))
        bert_batch, tokenized_abs, tokenized_ids = SentTripleBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
    
        # Get SEP indices from the sentences; some of the sentences may have been cut off
        # at some max length.
        num_abs_sents = []
        batch_sep_idxs = []
        max_num_sents = -1
        max_num_toks = -1
        for tokid_seq in tokenized_ids:
            sep_idxs = []
            for i, tok_id in enumerate(tokid_seq):
                if tok_id == pt_lm_tokenizer.sep_token_id:
                    sep_idxs.append(i)
            # Skip the first sep token because it is intended for the title sentence.
            sep_idxs = sep_idxs[1:]
            assert(len(sep_idxs) > 0)
            batch_sep_idxs.append(sep_idxs)
            num_sents = len(sep_idxs)
            num_abs_sents.append(num_sents)
            if num_sents > max_num_sents:
                max_num_sents = len(sep_idxs)
            if len(tokid_seq) > max_num_toks:
                max_num_toks = len(tokid_seq)
    
        # Pad the sep indices to max_num_sents and adjust them for max_num_toks
        flat_adjusted_sep_idxs = []
        for i, sep_idxs in enumerate(batch_sep_idxs):
            adjusted_idxs = [si+i*max_num_toks for si in sep_idxs]
            pad_len = max_num_sents - len(sep_idxs)
            # Pad so that indexing this gives the cls rep; which will be zeroed out eventually.
            adjusted_idxs.extend([0]*pad_len)
            flat_adjusted_sep_idxs.extend(adjusted_idxs)
    
        flat_sep_idxs = torch.LongTensor(flat_adjusted_sep_idxs)
    
        return bert_batch, batch_sep_idxs, num_abs_sents, flat_sep_idxs


class AbsSentTokBatcher(SentTripleBatcher):
    """
    Feeds a model which inputs query, positive and negative abstracts and sentence
    TOKEN indices for the abstracts. Negatives only at dev time, else the model uses in-batch
    negatives.
    """
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'query_abs_lens': list(int); Number of sentences in query abs.
                'query_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_abs_lens': list(int);
                'pos_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_abs_lens': list(int);
                'neg_senttok_idxs': list(list(list(int))); batch_size(
                    num_abs_sents(num_sent_tokens(ints)))
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, qabs_len, qabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
            query_texts, pt_lm_tokenizer)
        
        # Happens in the dev set.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, nabs_len, nabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
                neg_texts, pt_lm_tokenizer)
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
                pos_texts, pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs,
                'neg_bert_batch': nbert_batch, 'neg_abs_lens': nabs_len, 'neg_senttok_idxs': nabs_senttok_idxs
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
                pos_texts, pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch, 'abs_lens': qabs_len, 'senttok_idxs': qabs_senttok_idxs
            }
        return batch_dict
    
    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            seqs = [ex_abs['TITLE'] + ' [SEP] ']
            seqs.extend([s for s in ex_abs['ABSTRACT']])
            batch_abs_seqs.append(seqs)
        bert_batch, tokenized_abs, sent_token_idxs = AbsSentTokBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
        
        # Get SEP indices from the sentences; some of the sentences may have been cut off
        # at some max length.
        abs_lens = []
        for abs_sent_tok_idxs in sent_token_idxs:
            num_sents = len(abs_sent_tok_idxs)
            abs_lens.append(num_sents)
            assert (num_sents > 0)
            
        return bert_batch, abs_lens, sent_token_idxs

    @staticmethod
    def prepare_bert_sentences(sents, tokenizer):
        """
        Given a batch of documents with sentences prepare a batch which can be passed through BERT.
        Also keep track of the token indices for every sentence so sentence reps can be aggregated
        by averaging word embeddings.
        :param sents: list(list(string)); [batch_size[title and abstract sentences]]
        :param tokenizer: an instance of the appropriately initialized BERT tokenizer.
        :return:
        All truncated to max_num_toks by lopping off final sentence.
            bert_batch: dict(); bert batch.
            batch_tokenized_text: list(string); tokenized concated title and abstract.
            batch_sent_token_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
        """
        max_num_toks = 500
        # Construct the batch.
        tokenized_batch = []
        batch_tokenized_text = []
        batch_sent_token_idxs = []
        batch_seg_ids = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for abs_sents in sents:
            abs_tokenized_text = []
            abs_indexed_tokens = []
            abs_sent_token_indices = []  # list of list for every abstract.
            cur_len = 0
            for sent_i, sent in enumerate(abs_sents):
                tokenized_sent = tokenizer.tokenize(sent)
                # Convert token to vocabulary indices
                sent_indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
                # Add 1 for accounting for the CLS token which will be added
                # at the start of the sequence below.
                cur_sent_tok_idxs = [cur_len+i+1 for i in range(len(tokenized_sent))]
                # Store the token indices but account for the max_num_tokens
                if cur_len + len(cur_sent_tok_idxs) <= max_num_toks:
                    abs_sent_token_indices.append(cur_sent_tok_idxs)
                    abs_tokenized_text.extend(tokenized_sent)
                    abs_indexed_tokens.extend(sent_indexed_tokens)
                else:
                    len_exceded_by = cur_len + len(cur_sent_tok_idxs) - max_num_toks
                    reduced_len = len(cur_sent_tok_idxs) - len_exceded_by
                    # It can be that len_exceded_by is exactly len(cur_sent_tok_idxs)
                    # dont append a empty list then.
                    if reduced_len > 0:
                        abs_sent_token_indices.append(cur_sent_tok_idxs[:reduced_len])
                        abs_tokenized_text.extend(tokenized_sent[:reduced_len])
                        abs_indexed_tokens.extend(sent_indexed_tokens[:reduced_len])
                    break
                cur_len += len(cur_sent_tok_idxs)
            batch_tokenized_text.append(abs_tokenized_text)
            # Exclude the titles token indices.
            batch_sent_token_idxs.append(abs_sent_token_indices[1:])
            # Append CLS and SEP tokens to the text..
            abs_indexed_tokens = tokenizer.build_inputs_with_special_tokens(token_ids_0=abs_indexed_tokens)
            if len(abs_indexed_tokens) > max_seq_len:
                max_seq_len = len(abs_indexed_tokens)
            seq_lens.append(len(abs_indexed_tokens))
            tokenized_batch.append(abs_indexed_tokens)
            batch_seg_ids.append([0] * len(abs_indexed_tokens))
            batch_attn_mask.append([1] * len(abs_indexed_tokens))
        # Pad the batch.
        for ids_sent, seg_ids, attn_mask in zip(tokenized_batch, batch_seg_ids, batch_attn_mask):
            pad_len = max_seq_len - len(ids_sent)
            ids_sent.extend([tokenizer.pad_token_id] * pad_len)
            seg_ids.extend([tokenizer.pad_token_id] * pad_len)
            attn_mask.extend([tokenizer.pad_token_id] * pad_len)
        # The batch which the BERT model will input.
        bert_batch = {
            'tokid_tt': torch.tensor(tokenized_batch),
            'seg_tt': torch.tensor(batch_seg_ids),
            'attnmask_tt': torch.tensor(batch_attn_mask),
            'seq_lens': seq_lens
        }
        return bert_batch, batch_tokenized_text, batch_sent_token_idxs


class AbsSentTokBatcherPreAlign(AbsSentTokBatcher):
    """
    Feeds a model which inputs query, positive and negative abstracts and sentence
    TOKEN indices for the abstracts. Negatives only at dev time, else the model uses in-batch
    negatives.
    """
    # Which pre-aligned index to read. Can be: {'cc_align', 'abs_align'}
    align_type = 'cc_align'
    
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'query_abs_lens': list(int); Number of sentences in query abs.
                'query_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_abs_lens': list(int);
                'pos_align_idxs': list([int int]); query align sent idx, cand align sent idx
                'pos_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_abs_lens': list(int);
                'neg_align_idxs': list([int int]); query align sent idx, cand align sent idx
                'neg_senttok_idxs': list(list(list(int))); batch_size(
                    num_abs_sents(num_sent_tokens(ints)))
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, qabs_len, qabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
            query_texts, pt_lm_tokenizer)
        
        # Happens in the dev set.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, nabs_len, nabs_senttok_idxs, neg_align_idxs = AbsSentTokBatcherPreAlign.prepare_abstracts(
                neg_texts, pt_lm_tokenizer)
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs, pos_align_idxs = AbsSentTokBatcherPreAlign.prepare_abstracts(
                pos_texts, pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs,
                'neg_bert_batch': nbert_batch, 'neg_abs_lens': nabs_len, 'neg_senttok_idxs': nabs_senttok_idxs,
                'pos_align_idxs': pos_align_idxs, 'neg_align_idxs': neg_align_idxs
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs, pos_align_idxs = AbsSentTokBatcherPreAlign.prepare_abstracts(
                pos_texts, pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs,
                'pos_align_idxs': pos_align_idxs
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch, 'abs_lens': qabs_len, 'senttok_idxs': qabs_senttok_idxs
            }
        return batch_dict
    
    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
            pre_computed_alignments: list(list(int)); batch_size([q_idx, c_idx])
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        pre_computed_alignments = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            seqs = [ex_abs['TITLE'] + ' [SEP] ']
            seqs.extend([s for s in ex_abs['ABSTRACT']])
            batch_abs_seqs.append(seqs)
            if AbsSentTokBatcherPreAlign.align_type in ex_abs:
                assert(len(ex_abs[AbsSentTokBatcherPreAlign.align_type]) == 2)
                pre_computed_alignments.append(ex_abs[AbsSentTokBatcherPreAlign.align_type])
        bert_batch, tokenized_abs, sent_token_idxs = AbsSentTokBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
        
        # Get SEP indices from the sentences; some of the sentences may have been cut off
        # at some max length.
        abs_lens = []
        for abs_sent_tok_idxs in sent_token_idxs:
            num_sents = len(abs_sent_tok_idxs)
            abs_lens.append(num_sents)
            assert (num_sents > 0)
        
        if pre_computed_alignments:
            assert(len(pre_computed_alignments) == len(abs_lens))
            return bert_batch, abs_lens, sent_token_idxs, pre_computed_alignments
        else:
            return bert_batch, abs_lens, sent_token_idxs
