"""
Script to demo example usage of the Aspire Bi-encoder with linear mixing across
BERT layers, i.e the models: aspire-biencoder-biomed-scib-full,
aspire-biencoder-biomed-spec-full, and aspire-biencoder-compsci-spec-full.

*-all models released as zip folders alongside:
https://huggingface.co/allenai/aspire-biencoder-biomed-scib
https://huggingface.co/allenai/aspire-biencoder-biomed-spec
https://huggingface.co/allenai/aspire-biencoder-compsci-spec

Requirements:
- transformers version: 4.5.1
- torch version: 1.8.1

Code here is used here: https://github.com/allenai/aspire#specter-cocite
"""
import torch
from torch import nn as nn
from torch.nn import functional
from transformers import AutoModel, AutoTokenizer


# Define the linear mixing layer:
class SoftmaxMixLayers(torch.nn.Linear):
    def forward(self, input):
        # the weight vector is out_dim x in_dim.
        # so we want to softmax along in_dim.
        weight = functional.softmax(self.weight, dim=1)
        return functional.linear(input, weight, self.bias)


# Define the Aspire biencoder:
class AspireBiEnc(nn.Module):
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        self.bert_layer_weights = SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
    
    def forward(self, bert_batch):
        """
        Pass the title+abstract through BERT, read off CLS reps, and weighted combine across layers.
        """
        model_outputs = self.bert_encoder(**bert_batch)
        # Weighted combine the hidden_states which is a list of [bs x max_seq_len x bert_encoding_dim]
        # with as many tensors as layers + 1 input layer.
        hs_stacked = torch.stack(model_outputs.hidden_states, dim=3)
        weighted_sum_hs = self.bert_layer_weights(hs_stacked)  # [bs x max_seq_len x bert_encoding_dim x 1]
        weighted_sum_hs = torch.squeeze(weighted_sum_hs, dim=3)
        # Read of CLS token as document representation: (batch_size, sequence_length, hidden_size)
        cls_doc_reps = weighted_sum_hs[:, 0, :]
        return cls_doc_reps


