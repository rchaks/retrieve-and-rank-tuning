from __future__ import print_function

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict, OrderedDict
from copy import deepcopy

from rnr_debug_helpers.utils.answer import Answer
from rnr_debug_helpers.utils.io_helpers import LabelledQuery, initialize_logger, smart_file_open, \
    PredictionReader, initialize_query_stream
from rnr_debug_helpers.utils.stats import compute_average_precision_for_query, compute_ndcg_for_query, \
    compute_top_1_accuracy, compute_recall_for_query

MAX_PRECISION_THRESHOLDS = None
LOGGER = initialize_logger(logging.INFO, "Computing Accuracy")
_DEFAULT_SCORE = -sys.maxsize


def _get_next_n_scores_from_prediction_reader(predictions, num_to_read):
    """
    helper script to read a specific number of predictions from the predictions file

    :param RaasPredictionReader predictions: reader for raas prediction file
    :param num_to_read: number of answers to read
    :return: list of rank scores and list of confidence scores (if any confidence scores were provided)
    :rtype: tuple(list, list)
    """
    rank_scores_for_query = []
    if predictions.is_configured_with_confidence_scores():
        conf_scores_for_query = []
    else:
        conf_scores_for_query = None
    num_read = 0
    try:
        while num_read < num_to_read:
            p = next(predictions)
            num_read += 1
            rank_scores_for_query.append(p.rank_score)
            if conf_scores_for_query is not None:
                conf_scores_for_query.append(p.conf_score)
    except StopIteration:
        raise ValueError(
            "Expected to find at least %d more predictions to read, but only found %d" % (num_to_read, num_read))

    return rank_scores_for_query, conf_scores_for_query


def _update_ground_truth_answer_set_with_predictions(query, predictions, has_conf_scores):
    first_pass_candidate_answers = query.get_answer_ids()

    rank_scores_for_query = []
    if has_conf_scores:
        conf_scores_for_query = []
    else:
        conf_scores_for_query = None

    for answer_id in first_pass_candidate_answers:
        if answer_id in predictions:
            # we have a prediction - use it to score answer
            rank_scores_for_query.append(predictions[answer_id].rank_score)
            if conf_scores_for_query is not None:
                conf_scores_for_query.append(predictions[answer_id].conf_score)
        else:
            LOGGER.warn("No prediction found for answer id <<%s>>, defaulting scores to default score: %.4g" %
                        (answer_id, _DEFAULT_SCORE))
            rank_scores_for_query.append(_DEFAULT_SCORE)
            if conf_scores_for_query is not None:
                conf_scores_for_query.append(_DEFAULT_SCORE)
    query.set_rank_scores(rank_scores_for_query)
    query.set_conf_scores(conf_scores_for_query)


def _add_answers_that_were_predicted_but_not_in_ground_truth_set(query, predictions, has_conf_scores):
    answer_ids_from_ground_truth_file = query.get_answer_ids()
    for answer_id, prediction in predictions.iteritems():
        if answer_id not in answer_ids_from_ground_truth_file:
            # default the ground truth to 0 and add the answer
            if has_conf_scores:
                query.add_candidate_answer(feature_vector=[], label=0, rank_score=prediction.rank_score,
                                           confidence_score=prediction.conf_score, answer_id=answer_id)
            else:
                query.add_candidate_answer(feature_vector=[], label=0, rank_score=prediction.rank_score,
                                           confidence_score=None, answer_id=answer_id)


def update_query_with_prediction_scores(ground_truth_answers, prediction_reader):
    has_conf_scores = prediction_reader.is_configured_with_confidence_scores()

    if prediction_reader.is_configured_with_answer_ids():
        predictions = prediction_reader.get_all_predictions_till_next_query()
        _update_ground_truth_answer_set_with_predictions(ground_truth_answers, predictions, has_conf_scores)
        _add_answers_that_were_predicted_but_not_in_ground_truth_set(ground_truth_answers, predictions,
                                                                     has_conf_scores)
    else:
        # assume predictions are in the same order/quantity as query-doc pairs from ground truth set
        rank_scores_for_query, conf_scores_for_query = _get_next_n_scores_from_prediction_reader(
            prediction_reader, ground_truth_answers.get_answer_count())
        ground_truth_answers.set_rank_scores(rank_scores_for_query)
        ground_truth_answers.set_conf_scores(conf_scores_for_query)


def _get_top_ranked_answer(query):
    max_score = max(query.get_rank_scores())
    max_score_index = query.get_rank_scores().index(max_score)
    ans_id_of_max = query.get_answer_ids()[max_score_index]
    if query.get_conf_scores() is not None:
        conf_of_max = query.get_conf_score(ans_id_of_max)
    else:
        conf_of_max = max_score

    return Answer(query.get_qid(), ground_truth=query.get_label(ans_id_of_max), score=max_score,
                  answer_id=ans_id_of_max, confidence=conf_of_max)


def calc_stats_for_query(query, stats, stats_which_need_to_be_collected, k):
    for stat in stats_which_need_to_be_collected:
        if stat == 'average_precision_%d_truncated' % k:
            score = compute_average_precision_for_query(query, k=k)
        elif stat == 'ndcg@%d' % k:
            score = compute_ndcg_for_query(query, k=k)
        elif stat == 'top-1-accuracy':
            score = compute_top_1_accuracy(query)
        elif stat == 'recall@%d' % k:
            score = compute_recall_for_query(query, k=k)
        else:
            raise ValueError('Unsupported stat calculation: %s' % stat)
        stats[stat] += score
        LOGGER.debug("%s for query %s: %.4f" % (stat, query.get_qid(), score))


def update_with_confidence_score_based_stats(top_k_answer_set, stats, max_precision_thresholds, to_file=None):
    top_k_answer_set.sort(key=lambda x: x.confidence, reverse=True)

    # Initialize stats
    num_correct_so_far = 0
    num_incorrect_so_far = 0
    num_answered_so_far = 0
    precision = 0.0
    percent_answered = 0.0
    previous_threshold = float("inf")
    for threshold in max_precision_thresholds:
        stats["max_accuracy_at_precision_%s" % threshold] = 0.0

    # Setup a logger for the fine-grained stats per threshold
    try:
        if to_file is not None:
            to_file = smart_file_open(to_file, 'wb')
            writer = csv.writer(to_file)
        else:
            writer = csv.writer(sys.stdout)
        writer.writerow(["Confidence Threshold", "Percent Answered", "Precision", "Number Correct", "Number Incorrect"])

        # Do the updates by iterating through each possible threshold
        for answer in top_k_answer_set:
            if previous_threshold != answer.confidence:
                # print the values up to the previous threshold
                writer.writerow(
                    [previous_threshold, percent_answered, precision, num_correct_so_far, num_incorrect_so_far])
                previous_threshold = answer.confidence

            # Calculate stats up to current threshold
            num_answered_so_far += 1
            if _is_correct(answer):
                num_correct_so_far += 1
            else:
                num_incorrect_so_far += 1
            percent_answered = num_answered_so_far / float(len(top_k_answer_set))
            precision = num_correct_so_far / float(num_answered_so_far)

            # Update stats
            for threshold in max_precision_thresholds:
                if precision >= threshold:
                    accuracy_so_far = num_correct_so_far / float(len(top_k_answer_set))
                    stats["max_accuracy_at_precision_%s" % threshold] = accuracy_so_far

        # print the values at the last threshold
        writer.writerow([previous_threshold, percent_answered, precision, num_correct_so_far, num_incorrect_so_far])
    finally:
        if to_file is not None:
            to_file.close()


def _is_correct(answer):
    return answer.ground_truth > 0


def assign_labels_and_scores(predictions_by_answer_id, correct_answers_by_qid):
    qid = list(predictions_by_answer_id.values())[0].qid
    answer_ids = list()
    labels = list()
    rank_scores = list()
    min_rank_score = sys.float_info.max
    conf_scores = list()
    feature_vectors = list()

    if qid not in correct_answers_by_qid:
        raise ValueError('Qid: <<%s>> encountered in predictions file, but not in ground truth file' % qid)
    correct_answers_not_seen_yet = deepcopy(correct_answers_by_qid[qid])
    # Setup the predictions that were retrieved
    for answer_id, prediction in predictions_by_answer_id.items():
        answer_ids.append(answer_id)
        gt = 0
        if answer_id in correct_answers_not_seen_yet:
            gt = correct_answers_not_seen_yet.pop(answer_id)
        labels.append(gt)
        if prediction.rank_score < min_rank_score:
            min_rank_score = prediction.rank_score
        rank_scores.append(prediction.rank_score)
        conf_scores.append(prediction.conf_score)
        feature_vectors.append([0])

    # If there are correct answers that don't show up in teh retrieval results, append them at the end
    dummy_rank_score_for_answers_not_seen_in_list = min_rank_score - 100
    if correct_answers_not_seen_yet:
        for answer_id, label in correct_answers_not_seen_yet.items():
            answer_ids.append(answer_id)
            labels.append(label)
            rank_scores.append(dummy_rank_score_for_answers_not_seen_in_list)
            conf_scores.append(0)
            feature_vectors.append([0])
    return LabelledQuery(qid=qid, ans_ids=answer_ids, conf_scores=conf_scores, scores=rank_scores, labels=labels,
                         feature_vectors=feature_vectors)


def compute_performance_stats(ground_truth_query_stream, prediction_reader, k=5):
    stats_to_compute = ['ndcg@%d' % k, 'top-1-accuracy', 'average_precision_%d_truncated' % k,
                        'recall@%d' % k]
    top_one_answers = list()
    stats = defaultdict(int)
    LOGGER.info('first lookup ground truth, then iterate over prediction file')
    correct_answers_by_qid = generate_correct_answer_lookup(ground_truth_query_stream)
    stats['num_questions'] = len(correct_answers_by_qid)
    try:
        while True:
            query = assign_labels_and_scores(
                prediction_reader.get_all_predictions_till_next_query(),
                correct_answers_by_qid)
            stats['num_queries_predicted'] += 1
            stats['num_instances'] += query.get_answer_count()
            calc_stats_for_query(query, stats, stats_to_compute, k)
            top_one_answers.append(_get_top_ranked_answer(query))
    except StopIteration:
        # reached end of predictions
        stats['num_queries_skipped_in_preds_file'] = len(correct_answers_by_qid) - stats[
            'num_queries_predicted']

    for stat in stats_to_compute:
        stats[stat] /= float(stats['num_questions'])

    return stats, top_one_answers


def main(args):
    with smart_file_open(args.ground_truth_file) as ground_truth_file:
        LOGGER.info('Reading in the ground truth first from input file: %s' % ground_truth_file)
        query_stream = initialize_query_stream(ground_truth_file, args.infile_format)

        with smart_file_open(args.prediction_file) as preds_file:
            LOGGER.info('Reading predictions from file: %s' % args.prediction_file)
            prediction_reader = PredictionReader(preds_file, file_has_answer_ids=True,
                                                 file_has_confidence_scores=args.file_has_confidence_scores)
            stats, top_one_answers = compute_performance_stats(query_stream, prediction_reader, k=args.top_k)

    if args.accuracy_vs_qa_file is not None:
        update_with_confidence_score_based_stats(top_one_answers, stats, parsed_args.max_acc_prec_thresholds,
                                                 args.accuracy_vs_qa_file)
    else:
        update_with_confidence_score_based_stats(top_one_answers, stats, parsed_args.max_acc_prec_thresholds)

    json.dump(stats, sys.stdout, sort_keys=True, indent=4)
    if args.accuracy_file is not None:
        with smart_file_open(args.accuracy_file, 'w') as outfile:
            json.dump(stats, outfile, sort_keys=True, indent=4)


def generate_correct_answer_lookup(queries):
    answer_label_lookup = OrderedDict()
    stats = defaultdict(int)
    for query in queries:
        stats['num_questions'] += 1
        stats['num_instances'] += query.get_answer_count()
        answer_label_lookup['%s' % stats['num_questions']] = {aid: query.get_label(aid) for aid in
                                                              query.get_answer_ids() if
                                                              query.get_label(aid) > 0}
        stats['num_possible_correct'] += len(answer_label_lookup['%s' % stats['num_questions']])

        LOGGER.debug('Read ground truth for %d queries' % stats['num_questions'])
    LOGGER.info('Done reading in ground truth')
    json.dump(stats, sys.stdout, sort_keys=True, indent=4)
    return answer_label_lookup


if __name__ == '__main__':
    # Get cmd line args
    parser = argparse.ArgumentParser(
        description="Evaluates ranker predictions", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--predictionFile', dest='prediction_file', required=True,
                        help="file containing predictions (in a space delimited format with <qid> <aid> <rank_score> "
                             "(<conf_score>)")
    parser.add_argument('-c', '--confidenceScores',
                        help="Optionally indicate that the prediction file contains a column of confidence scores",
                        action="store_const",
                        dest="file_has_confidence_scores", const=True, default=False)
    parser.add_argument('-k', '--topK', default=10, type=int, dest="top_k",
                        help="Optionally indicate a different top k setting for ndcg and average precision truncation")
    parser.add_argument('-g', '--groundTruthFile', dest='ground_truth_file', required=True,
                        help="a file containing ground_truth")
    parser.add_argument('-f', '--file_format', dest='infile_format', choices=['relevance_file', 'rnr_tooling_export'],
                        default='relevance_file',
                        help="indicate whether the input file is in the `relevance file` format that's described in "
                             "the [Training a ranker by using the train.py script](https://www.ibm.com/watson/devel"
                             "opercloud/doc/retrieve-rank/training_data.html#script) or the `export-questions.json` "
                             "file that can be exported from the RnR Tooling in the [Training a ranker by using the "
                             "train.py script](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/training_"
                             "data.html#script)"),
    parser.add_argument('-d', '--debug', help="Optionally print lots of debugging statements", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-a', '--accuracyFile', dest='accuracy_file', required=False,
                        help="Optionally generate a json file with the ranking performance (i.e. accuracy, precision..."
                             ") stats (otherwise just prints these to stdout)")
    parser.add_argument('-m', '--makeTable', dest='accuracy_vs_qa_file',
                        help="Optionally generate a csv file containing Percent Questions Answered vs Precision values "
                             "at various confidence thresholds")
    parser.add_argument('-t', '--max_accuracy_at_precision_thresholds', dest='max_acc_prec_thresholds', type=float,
                        nargs="+", default=[0.2, 0.7],
                        help="Precision thresholds at which to calculate max possible accuracy using confidence "
                             "scores. To generate a table, can set all these thresholds: [0.05, 0.1, 0.15,..., 1]")

    parsed_args = parser.parse_args()

    LOGGER.setLevel(parsed_args.loglevel)
    # setup constants

    main(parsed_args)
