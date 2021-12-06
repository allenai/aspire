"""
Models which learn sentence representations.
Mostly a bunch of wrappers for raw bert models which are finetuned.
"""
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel


class SentBERTWrapper(nn.Module):
    """
    Pass sentence through encoder and minimize triple loss
    with inbatch negatives.
    """
    
    def __init__(self, model_name):
        """
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.sent_encoder = AutoModel.from_pretrained(model_name)
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')

    def forward(self, batch_dict):
        batch_bpr = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_bpr
        }
        return loss_dict
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
            }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']
        # Get the representations from the model.
        q_sent_reps = self.sent_reps_bert(bert_model=self.sent_encoder, bert_batch=qbert_batch)
        p_context_reps = self.sent_reps_bert(bert_model=self.sent_encoder, bert_batch=pbert_batch)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            n_context_reps = self.sent_reps_bert(bert_model=self.sent_encoder, bert_batch=nbert_batch)
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]
        loss_val = self.criterion(q_sent_reps, p_context_reps, n_context_reps)
        return loss_val

    @staticmethod
    def sent_reps_bert(bert_model, bert_batch):
        """
        Pass the concated abstract through BERT, and read off [SEP] token reps to get sentence reps,
        and weighted combine across layers.
        :param bert_model: torch.nn.Module subclass. A bert model.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = bert_model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        cls_doc_reps = model_outputs.last_hidden_state[:, 0, :]
        return cls_doc_reps.squeeze()


class ICTBERTWrapper(SentBERTWrapper):
    """
    Pass sentence through encoder and minimize triple loss
    with inbatch negatives.
    """
    
    def __init__(self, model_name):
        """
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.sent_encoder = AutoModel.from_pretrained(model_name)
        self.context_encoder = AutoModel.from_pretrained(model_name)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
            }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']
        # Get the representations from the model.
        q_sent_reps = self.sent_reps_bert(bert_model=self.sent_encoder, bert_batch=qbert_batch)
        p_context_reps = self.sent_reps_bert(bert_model=self.context_encoder, bert_batch=pbert_batch)
        batch_size = q_sent_reps.size(0)
        assert(q_sent_reps.size(1) == p_context_reps.size(1) == self.bert_encoding_dim)
        # Get similarities from query sent reps to all contexts (non pos ones are inbatch negs).
        dot_sims = torch.matmul(q_sent_reps, p_context_reps.T)
        assert(dot_sims.size(0) == dot_sims.size(1) == batch_size)
        # Correct context targets are just the corresponding ids for every element.
        targets = torch.arange(batch_size)
        targets = Variable(targets)
        if torch.cuda.is_available():
            targets = targets.cuda()
        loss_val = self.criterion(dot_sims, targets)
        return loss_val
