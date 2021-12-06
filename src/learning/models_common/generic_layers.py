"""
Generic layers used across models.
"""
import torch
from torch import nn as nn
from torch.nn import functional
import collections
from . import activations

non_linearities = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'softplus': torch.nn.Softplus
}


class FeedForwardNet(nn.Module):
    def __init__(self, in_dim, out_dim, non_linearity,
                 ffn_composition_dims=None, dropoutp=0.3, use_bias=True, score_ffn=False):
        """
        :param in_dim: int; dimension of input to ffn.
        :param ffn_composition_dims: tuple(int); hidden layers dimensions for the classifier.
        :param out_dim: int; dimensions of output from ffn.
        :param dropoutp: float; dropout probability.
        :param non_linearity: string; non-lin after linear layer.
        :param use_bias: bool; says if linear layer should have a bias.
        :param score_ffn: bool; says if the final layer output is an attention score - if so,
            doesnt apply a non-linearity on it.
        """
        torch.nn.Module.__init__(self)
        # Layers of the feed-forward network.
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = collections.OrderedDict()
        if ffn_composition_dims:
            # Concat the dimensionality of the output layer
            ffn_composition_dims = ffn_composition_dims + (out_dim,)
            layers['lin_0'] = torch.nn.Linear(in_features=in_dim,
                                              out_features=ffn_composition_dims[0], bias=use_bias)
            layers['nonlin_0'] = non_linearities[non_linearity]()
            layers['dropout_0'] = torch.nn.Dropout(p=dropoutp)
            for layer_i in range(len(ffn_composition_dims) - 1):
                layers['lin_{:d}'.format(layer_i + 1)] = \
                    torch.nn.Linear(in_features=ffn_composition_dims[layer_i],
                                    out_features=ffn_composition_dims[layer_i + 1],
                                    bias=use_bias)
                # If its a score ffn then dont add a non-linearity at the final layer.
                if layer_i == len(ffn_composition_dims) - 2 and score_ffn:
                    assert(ffn_composition_dims[layer_i + 1] == 1)
                    pass
                else:
                    layers['nonlin_{:d}'.format(layer_i + 1)] = non_linearities[non_linearity]()
                # Dont add dropout at the final layer.
                if layer_i != len(ffn_composition_dims) - 2:
                    layers['dropout_{:d}'.format(layer_i + 1)] = torch.nn.Dropout(p=dropoutp)
        else:
            layers['lin_0'] = torch.nn.Linear(in_features=self.in_dim,
                                              out_features=out_dim, bias=use_bias)
            layers['nonlin_0'] = non_linearities[non_linearity]()
        self.ffn = nn.Sequential(layers)

    def forward(self, in_feats):
        """
        :param in_feats: torch.Tensor(batch_size, in_dim)
        :return: out_feats: torch.Tensor(batch_size, out_dim)
        """
        return self.ffn.forward(in_feats)


class SoftmaxMixLayers(torch.nn.Linear):
    """
    Combine bert representations across layers with a weighted sum
    where the weights are softmaxes over a set of learned parameters.
    """
    def forward(self, input):
        # the weight vector is out_dim x in_dim.
        # so we want to softmax along in_dim.
        weight = functional.softmax(self.weight, dim=1)
        return functional.linear(input, weight, self.bias)


class GatedAttention(nn.Module):
    """
    Implements the gated attention in:
    Attention-based Deep Multiple Instance Learning
    http://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf
    """
    def __init__(self, embed_dim):
        torch.nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.internal_dim = embed_dim
        self.lin_V = nn.Linear(in_features=embed_dim, out_features=self.internal_dim, bias=False)
        self.V_nonlin = nn.Tanh()
        self.lin_U = nn.Linear(in_features=embed_dim, out_features=self.internal_dim, bias=False)
        self.gate_sigm = nn.Sigmoid()
        self.score_weight = nn.Linear(in_features=embed_dim, out_features=1, bias=False)
    
    def forward(self, in_seq, seq_lens):
        """
        :param in_seq: torch.tensor; batch_size x max_seq_len x embed_dim
        :param seq_lens: list(int); batch_size
        :return attention_weights: batch_size x max_seq_len
        """
        batch_size, max_seq_len = in_seq.size(0), in_seq.size(1)
        in_seq = in_seq.view(batch_size*max_seq_len, self.embed_dim)
        # batch_size*max_seq_len x internal_dim
        hidden = self.V_nonlin(self.lin_V(in_seq))
        gates = self.gate_sigm(self.lin_U(in_seq))
        scores = self.score_weight(hidden*gates).squeeze()
        scores = scores.view(batch_size, max_seq_len)
        # This expects the padded elements to be zero.
        attention_weights = activations.masked_softmax(batch_scores=scores, target_lens=seq_lens)
        return attention_weights

# Straught through estimator from: https://www.hassanaskary.com/python/pytorch/
# deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return functional.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    
    def forward(self, x):
        x = STEFunction.apply(x)
        return x
