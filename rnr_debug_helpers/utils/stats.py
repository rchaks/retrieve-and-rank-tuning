import math
from warnings import warn

_NEGATIVE_INFINITY = float("-inf")
_POSITIVE_INFINITY = float("inf")
_EPSILON = 1E-8


def average_precision(ids_for_correct_answers, predicted_answer_id_sequence, k=10):
    """
    Adpated from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    Computes the average precision at k. This function computes the average prescision at k between two lists of
    items.

    :param list ids_for_correct_answers: A list of elements that are to be predicted (order doesn't matter)
    :param list predicted_answer_id_sequence: A list of predicted elements (order does matter)
    :param int k: The maximum number of predicted elements. This is optional, defaults to 10.
    :return: The average precision at k over the input lists
    :rtype: float
    """

    if len(predicted_answer_id_sequence) > k:
        predicted_answer_id_sequence = predicted_answer_id_sequence[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted_answer_id_sequence):
        if p in ids_for_correct_answers and p not in predicted_answer_id_sequence[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not ids_for_correct_answers:
        return 0.0

    return score / min(len(ids_for_correct_answers), k)


def ndcg(rank_order, ideal_order, k=5):
    """
    This is a function to get ndcg
    :param list rank_order: sequence of ground truth in the rank order returned by the system
    :param list ideal_order: sequence of ground truth in the ideal order
    :param int k: top k at which to evaluate ndcg. Optional, defaults to 5
    :return: ndcg for this ordering of ground truth
    :rtype: numeric
    """
    ideal_ndcg = _compute_dcg(ideal_order, k)
    if ideal_ndcg == 0:
        ndcg = 0.0
    else:
        ndcg = _compute_dcg(rank_order, k) / float(ideal_ndcg)
    return ndcg


def _compute_dcg(s, k):
    """
    A function to compute dcg
    :param s: sequence of ground truth in the rank order to use for calculating dcg
    :param k: top k at which to evaluate ndcg
    :return: dcg for this ordering of ground truth
    :rtype: numeric
    """
    dcg = 0.0
    for i in range(min(k, len(s))):
        dcg += (math.pow(2, s[i]) - 1) / math.log(i + 2, 2)
    return dcg


def min_max_norm(scores):
    """
    Provide min-max normalized set of scores
    :param list(numeric) scores: list of raw scores
    :return: list of normalized scores
    :rtype: list(numeric)
    """
    if len(scores) < 0:
        raise ValueError("No scores")
    elif len(scores) == 1:
        return [0]
    else:
        min_score = _POSITIVE_INFINITY
        max_score = _NEGATIVE_INFINITY
        for s in scores:
            if s < min_score:
                min_score = s
            if s > max_score:
                max_score = s
        normalizing_constant = max_score - min_score
        if normalizing_constant == 0:
            normalizing_constant = 1

        norm_scores = list()
        for s in scores:
            norm_scores.append(float(s - min_score) / normalizing_constant)
        return norm_scores


def sum_norm_scores(scores):
    """
    normalizes scores by dividing them by the sum of the scores
    :param list(numeric) scores: scores to normalize.  Assumes scores are all positive
    :return: normalized scores
    :rtype: list(numeric)
    """
    if len(scores) < 0:
        raise ValueError("No scores")
    elif len(scores) == 1:
        return [0]
    else:
        sum_scores = float(sum(scores))
        if sum_scores > _EPSILON:
            return [s / sum_scores for s in scores]
        else:
            warn("Unable to normalize using distribution sum, because sum is too close to zero: %.4f" % sum_scores)
            return scores


def compute_entropy(scores, is_pre_normalized=False, log_base=None):
    """
    Computes entropy of the scores, prenormalizing if required
    :param list(numeric) scores: original scores
    :param bool is_pre_normalized: if not True, then calls min_max_norm followed by sum norm (effectively achieves
        dist weighting on the scores after normalizing them between 0 and 1)
    :param int log_base: an optional different log base to use
    :return: entropy score for this distribution of scores
    :rtype: float
    """
    if not is_pre_normalized:
        scores = sum_norm_scores(min_max_norm(scores))

    entropy = 0.0
    for s in scores:
        if abs(s) < _EPSILON:
            entropy += s
        else:
            entropy += s * math.log(s)

    if log_base is not None:
        entropy /= math.log(log_base)

    return -1.0 * entropy


def compute_top_1_accuracy(query):
    """
    Compute the top 1 error for this query results.  Implementation note: returns 1/0 regardless of whether or not the
    label is on a multi-relevance scale.

    :param LabelledQuery query: query with updated doc ids, rank scores and ground truth for all candidate docs
    :return: 1 if ground truth for top-1 ranked answer > 0, 0 otherwise
    :rtype: int
    """
    ground_truth_for_max = None
    max_score = _NEGATIVE_INFINITY
    for g, r in zip(query.get_labels(), query.get_rank_scores()):
        if r > max_score:
            max_score = r
            ground_truth_for_max = g
    if ground_truth_for_max > 0:
        top_1_accuracy = 1
    else:
        top_1_accuracy = 0

    return top_1_accuracy


def get_ground_truth_ordering_by_rank_score(query):
    return [i for i, s in sorted(zip(query.get_labels(), query.get_rank_scores()), key=lambda pair: pair[1],
                                 reverse=True)]


def compute_ndcg_for_query(query, k=10):
    """
    Compute the ndcg for this query results.  Implementation note: truncates at min(num_docs,
    k).

    :param LabelledQuery query: query with updated doc ids, rank scores and ground truth for all candidate docs
    :param int k: k for truncation
    :return: ndcg
    :rtype: float
    """
    # Get the ground truth ordering from the candidate answer ranks
    ground_truth_ordering_by_rank_score = get_ground_truth_ordering_by_rank_score(query)
    ideal_ground_truth_ordering = sorted(query.get_labels(), reverse=True)
    ndcg_for_query = ndcg(rank_order=ground_truth_ordering_by_rank_score, ideal_order=ideal_ground_truth_ordering,
                          k=k)
    return ndcg_for_query


def compute_recall_for_query(query, k=10):
    """
    Compute the recall for this query results.  Implementation note: truncates at min(num_docs,
    k).

    :param LabelledQuery query: query with updated doc ids, rank scores and ground truth for all candidate docs
    :param int k: k for truncation
    :return: recall
    :rtype: float
    """
    # Get the ground truth ordering from the candidate answer ranks
    num_possible_correct = len([l for l in query.get_labels() if l > 0])
    if num_possible_correct > 0:
        ground_truth_ordering_by_rank_score = get_ground_truth_ordering_by_rank_score(query)
        num_correct_in_topk = len([l for l in ground_truth_ordering_by_rank_score[:k] if l > 0])
        recall = float(num_correct_in_topk) / num_possible_correct
    else:
        # treat this edge case as perfect score since no false negatives
        recall = 1.0
    return recall


def compute_average_precision_for_query(query, k=10):
    """
    Compute the average precision for this query results.  Implementation note: truncates at min(num_possible_correct,
    K_FOR_PRECISION_TRUNCATION).

    :param LabelledQuery query: query with updated doc ids, rank scores and ground truth for all candidate docs
    :param int k: k for truncation
    :return: average precision
    :rtype: float
    """
    answer_ids_that_are_correct = list()
    for ans_id in query.get_answer_ids():
        if query.get_label(ans_id) > 0:
            answer_ids_that_are_correct += [ans_id]

    answer_ids_sorted_by_rank_score = [i for i, s in sorted(zip(query.get_answer_ids(), query.get_rank_scores()),
                                                            key=lambda pair: pair[1], reverse=True)]
    ap = average_precision(answer_ids_that_are_correct, answer_ids_sorted_by_rank_score, k=k)

    return ap