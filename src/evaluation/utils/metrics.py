"""
From: https://gist.github.com/bwhite/3726239#file-rank_metrics-py
"""
import numpy as np


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)

    Relevance is binary (nonzero is relevant).

    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision

    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    "An Introduction to Neural Information Retrieval" writes method 1 for DCG
    so using that as default.
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(ranked_rel, atk, max_total_relevant):
    """
    Compute recall at k.
    :param ranked_rel: list(int); ranked list of relevance judged data.
    :param atk: int; rank at which to compute metric.
    :param max_total_relevant: int; maximum relevant to consider in
        case there are more relevant in total.
    :return: recall: float.
    """
    total_relevant = sum(ranked_rel)
    total_relevant = min(max_total_relevant, total_relevant)
    relatk = sum(ranked_rel[:atk])
    if total_relevant > 0:
        recall_atk = float(relatk)/total_relevant
    else:
        recall_atk = 0.0
    return recall_atk

def compute_metrics(ranked_judgements, pr_atks, threshold_grade):
    """
    Given the ranked judgements compute precision, recall and f1.
    :param ranked_judgements: list(int); graded or binary relevances in rank order.
    :param pr_atks: list(int); the @K values to use for computing precision and recall.
    :param threshold_grade: int; Assuming 0-3 graded relevances, threshold at some point
        and convert graded to binary relevance. If passed also compute NDCG.
    :return:
    """
    metrics = {}
    graded_judgements = ranked_judgements
    ranked_judgements = [1 if rel >= threshold_grade else 0 for rel in graded_judgements]
    # Use the full set of candidate not the pr_atk.
    ndcg = ndcg_at_k(graded_judgements, len(ranked_judgements))
    ndcg_20 = ndcg_at_k(graded_judgements, 20)
    ndcg_50 = ndcg_at_k(graded_judgements, 50)
    for atk in [5, 10, 15, 20, 25]:
        ndcg_pr_atk = ndcg_at_k(graded_judgements, int((atk/100)*len(ranked_judgements)))
        metrics[f'ndcg%{atk}'] = float(ndcg_pr_atk)
    max_total_relevant = sum(ranked_judgements)
    for atk in pr_atks:
        recall = recall_at_k(ranked_rel=ranked_judgements,
                             atk=atk, max_total_relevant=max_total_relevant)
        precision = precision_at_k(r=ranked_judgements, k=atk)
        f1 = 2*precision*recall/(precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f'precision@{atk}'] = float(precision)
        metrics[f'recall@{atk}'] = float(recall)
        metrics[f'f1@{atk}'] = float(f1)
    r_precision_ = r_precision(r=ranked_judgements)
    av_precision = average_precision(r=ranked_judgements)
    reciprocal_rank = mean_reciprocal_rank(rs=[ranked_judgements])
    metrics['r_precision'] = float(r_precision_)
    metrics['av_precision'] = float(av_precision)
    metrics['reciprocal_rank'] = float(reciprocal_rank)
    metrics['ndcg'] = float(ndcg)
    metrics['ndcg@20'] = float(ndcg_20)
    metrics['ndcg@50'] = float(ndcg_50)
    return metrics
