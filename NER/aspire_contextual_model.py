import torch
from torch import nn
from transformers import AutoModel
import numpy as np
from torch.autograd import Variable

# Define the Aspire contextual encoder with embeddings:
class AspireConSenContextual(nn.Module):
    def __init__(self, hf_model_name):
        """
        :param hf_model_name: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(hf_model_name)
        self.bert_encoder.config.output_hidden_states = True

    def forward(self, bert_batch, abs_lens, sent_tok_idxs, ner_tok_idxs):
        """
        Pass a batch of sentences through BERT and get sentence
        reps based on averaging contextual token embeddings.
        :return:
            sent_reps: batch_size x num_sents x encoding_dim
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.consent_reps_bert(bert_batch=bert_batch,
                                                         batch_senttok_idxs=sent_tok_idxs,
                                                         batch_nertok_idxs=ner_tok_idxs,
                                                         num_sents=abs_lens)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)

        return doc_cls_reps, sent_reps

    def _get_sent_reps(self,
                       final_hidden_state,
                       batch_tok_idxs,
                       batch_size,
                       max_sents,
                       max_seq_len):
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
            # Build a mask for the ith sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_tok_idxs[batch_abs_i]
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
            cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        return sent_reps

    def _get_ner_reps(self, final_hidden_state, batch_nertok_idxs):
        ner_reps = []
        for i, abs_toks in enumerate(batch_nertok_idxs):
            for ner_toks in abs_toks:
                tokens_for_ner = final_hidden_state[i, ner_toks]
                ner_rep = tokens_for_ner.mean(dim=0)[None, None, :]
                ner_reps.append(ner_rep)
        return ner_reps


    def consent_reps_bert(self, bert_batch, batch_senttok_idxs, batch_nertok_idxs, num_sents):
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
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = self._get_sent_reps(final_hidden_state, batch_senttok_idxs, batch_size, max_sents, max_seq_len)
        # Do the same for all ners in each sentence to get ner representation
        ner_reps = self._get_ner_reps(final_hidden_state, batch_nertok_idxs)
        final_reps = torch.cat(sent_reps + ner_reps, dim=1)
        return doc_cls_reps, final_reps
