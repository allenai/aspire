"""
Script to demo example usage of the Aspire Multi-Vector encoder which
represents documents via contextual sentence embeddings, i.e the models:
allenai/aspire-contextualsentence-singlem-biomed and
allenai/aspire-contextualsentence-singlem-compsci

Models released at:
https://huggingface.co/allenai/aspire-contextualsentence-singlem-biomed
https://huggingface.co/allenai/aspire-contextualsentence-singlem-compsci

Requirements:
- transformers version: 4.5.1
- torch version: 1.8.1
"""
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer


# Define the Aspire contextual encoder:
class AspireConSent(nn.Module):
    def __init__(self, hf_model_name):
        """
        :param hf_model_name: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(hf_model_name)
        self.bert_encoder.config.output_hidden_states = True

    def forward(self, bert_batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and get sentence
        reps based on averaging contextual token embeddings.
        :return:
            sent_reps: batch_size x num_sents x encoding_dim
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.consent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                         batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
        
        return doc_cls_reps, sent_reps

    def consent_reps_bert(self, bert_batch, batch_senttok_idxs, num_sents):
        """
        Pass the concated abstract through BERT, and average token reps to get contextual sentence reps.
        -- NO weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :param batch_senttok_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
        :param num_sents: list(int); number of sentences in each example in the batch passed.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
            sent_reps: FloatTensor [batch_size x num_sents x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        max_sents = max(num_sents)
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        # if torch.cuda.is_available():
        #     tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
            # Build a mask for the ith sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
                try:
                    sent_i_tok_idxs = abs_sent_idxs[sent_i]
                except IndexError:  # This happens in the case where the abstract has fewer than max sents.
                    sent_i_tok_idxs = []
                cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask))
            # if torch.cuda.is_available():
            #     sent_mask = sent_mask.cuda()
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1)/ \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_cls_reps, sent_reps


# Both below functions copied over from src.learning.batchers
# Function to prepare tokenize, pad inputs, while maintaining token indices
# for getting contextual sentence eocndings.
def prepare_bert_sentences(batch_doc_sents, tokenizer):
    """
    Given a batch of documents with sentences prepare a batch which can be passed through BERT.
    And keep track of the token indices for every sentence so sentence reps can be aggregated
    by averaging word embeddings.
    :param batch_doc_sents: list(list(string)); [batch_size[title and abstract sentences]]
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
    for abs_sents in batch_doc_sents:
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


# Prepare a batch of abstracts for passing through the model.
def prepare_abstracts(batch_abs, pt_lm_tokenizer):
    """
    Given the abstracts sentences as a list of strings prep them to pass through model.
    :param batch_abs: list(dict); list of example dicts with abstract sentences, and titles.
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
    bert_batch, tokenized_abs, sent_token_idxs = prepare_bert_sentences(
        batch_doc_sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
    
    # Get SEP indices from the sentences; some of the sentences may have been cut off
    # at some max length.
    abs_lens = []
    for abs_sent_tok_idxs in sent_token_idxs:
        num_sents = len(abs_sent_tok_idxs)
        abs_lens.append(num_sents)
        assert (num_sents > 0)
    
    return bert_batch, abs_lens, sent_token_idxs
