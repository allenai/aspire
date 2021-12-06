"""
Functions used across models.
"""
import numpy as np
import torch
from torch.nn import functional
from torch.autograd import Variable


def masked_softmax(batch_scores, target_lens):
    """
    Given the scores for the assignments for every example in the batch apply
    a masked softmax for the variable number of assignments.
    :param batch_scores: torch Tensor; batch_size x max_num_asgns; With non target
        scores set to zero.
    :param target_lens: list(int) [batch_size]; number of elemenets over which to
        compute softmax in each example of the batch.
    :return: probs: torch Tensor; same size as batch_scores.
    """
    batch_size, max_num_targets = batch_scores.size()
    # Set all the logits beyond the targets to very large negative values
    # so they contribute minimally to the softmax.
    logit_mask = np.zeros((batch_size, max_num_targets))
    for i, len in enumerate(target_lens):
        logit_mask[i, len:] = -1e32
    logit_mask = Variable(torch.FloatTensor(logit_mask))
    if torch.cuda.is_available():
        logit_mask = logit_mask.cuda()
    # Work with log probabilities because its the numerically stable softmax.
    batch_scores = batch_scores + logit_mask
    log_probs = functional.log_softmax(batch_scores, dim=1)
    return log_probs.exp()


def masked_2d_softmax(batch_scores, target_lens1, target_lens2):
    """
    Given the scores for the assignments for every example in the batch apply
    a masked softmax for the variable number of assignments.
    :param batch_scores: torch Tensor; batch_size x dim1 x dim2; With non target
        scores set to zero.
    :param target_lens1: list(int) [batch_size]; number of elemenets over which to
        compute softmax in each example of the batch along dim 1.
    :param target_lens2: list(int) [batch_size]; number of elemenets over which to
        compute softmax in each example of the batch along dim 2.
    :return: probs: torch Tensor; same size as batch_scores.
    """
    batch_size, q_max_size, c_max_size = batch_scores.size()
    # Set all the logits beyond the targets to very large negative values
    # so they contribute minimally to the softmax.
    logit_mask = np.zeros((batch_size, q_max_size, c_max_size))
    for i, (len1, len2) in enumerate(zip(target_lens1, target_lens2)):
        logit_mask[i, len1:, :] = -1e32
        logit_mask[i, :, len2:] = -1e32
    logit_mask = Variable(torch.FloatTensor(logit_mask))
    if torch.cuda.is_available():
        logit_mask = logit_mask.cuda()
    # Work with log probabilities because its the numerically stable softmax.
    batch_scores = batch_scores + logit_mask
    log_probs = functional.log_softmax(batch_scores.view(batch_size, q_max_size*c_max_size), dim=1)
    log_probs = log_probs.view(batch_size, q_max_size, c_max_size)
    return log_probs.exp()
