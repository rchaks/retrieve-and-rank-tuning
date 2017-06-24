import argparse
import csv
import logging
import sys
from collections import defaultdict
from os import path

from rnr_debug_helpers.compute_ranking_stats import generate_correct_answer_lookup, assign_labels_and_scores
from rnr_debug_helpers.utils.io_helpers import initialize_logger, smart_file_open, \
    PredictionReader, initialize_query_stream
from rnr_debug_helpers.utils.stats import compute_recall_for_query

LOGGER = initialize_logger(logging.INFO, path.basename(__file__))


def compute_recall_stats(k_settings_for_recall, labelled_query_stream, prediction_reader):
    correct_answers_by_qid = generate_correct_answer_lookup(labelled_query_stream)

    LOGGER.info("scoring predictions from: %s (against labels from %s)" % (prediction_reader, labelled_query_stream))
    stats = defaultdict(int)
    stats['num_queries'] = len(correct_answers_by_qid)
    try:
        while True:
            labelled_answer_set = assign_labels_and_scores(prediction_reader.get_all_predictions_till_next_query(),
                                                           correct_answers_by_qid)
            stats['num_queries_predicted'] += 1
            for k_for_recall in k_settings_for_recall:
                stats['recall@%d' % k_for_recall] += compute_recall_for_query(labelled_answer_set,
                                                                              k_for_recall)
                LOGGER.debug("query: %s, recall@%d: %.4f" %
                             (labelled_answer_set.get_qid(), k_for_recall,
                              stats['recall@%d' % k_for_recall]))
    except StopIteration:
        # reached end of predictions
        stats['num_queries_skipped_in_preds_file'] = len(correct_answers_by_qid) - stats[
            'num_queries_predicted']
    _average_recall_across_queries(stats, k_settings_for_recall)

    return {k: stats['recall@%d' % k] for k in k_settings_for_recall}


def _average_recall_across_queries(stats, k_settings_for_recall):
    for k in k_settings_for_recall:
        recall = float(stats['recall@%d' % k])
        # calculate across all queries
        stats['recall@%d' % k] = recall / stats['num_queries']
        # calculate only on predicted
        stats['recall_on_predicted_only@%d' % k] = recall / stats['num_queries_predicted']

    return stats


def print_recall_stats_to_csv(recall_stats, outfile):
    writer = csv.writer(outfile)
    writer.writerow(['K', 'Recall@K'])
    for k in sorted(recall_stats.keys()):
        writer.writerow([k, recall_stats[k]])


def main(args):
    LOGGER.info('Start Script')
    with smart_file_open(args.ground_truth_file) as infile:
        ground_truth_query_stream = initialize_query_stream(infile, file_format=args.infile_format)
        with smart_file_open(args.prediction_file) as preds_file:
            prediction_reader = PredictionReader(preds_file, file_has_confidence_scores=args.file_has_confidence_scores)
            recall_stats = compute_recall_stats(args.recall_settings, ground_truth_query_stream, prediction_reader)

    print_recall_stats_to_csv(recall_stats, args.output_file)

    LOGGER.info('Great Success')


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
    parser.add_argument('recall_settings', metavar='N', type=int, nargs='+',
                        help='a list of settings at which recall will be calculated (must be integers)')
    parser.add_argument('-g', '--groundTruthFile', dest='ground_truth_file', required=True,
                        help="a file containing ground_truth")
    parser.add_argument('-f', '--file_format', dest='infile_format', choices=['relevance_file', 'rnr_tooling_export'],
                        default='relevance_file',
                        help="indicate whether the input file is in the `relevance file` format that's described in "
                             "the [Training a ranker by using the train.py script](https://www.ibm.com/watson/devel"
                             "opercloud/doc/retrieve-rank/training_data.html#script) or the `export-questions.json` "
                             "file that can be exported from the RnR Tooling in the [Training a ranker by using the "
                             "train.py script](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/ranker_"
                             "tooling.html)"),
    parser.add_argument('-d', '--debug', help="Optionally print lots of debugging statements", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-o', '--outputFile', dest='output_file', default=sys.stdout, type=argparse.FileType('wb'),
                        help="Optionally generate a csv file with the recall calculations at each k setting."
                             " If no file is provided, it'll print to stdout")

    parsed_args = parser.parse_args()

    LOGGER.setLevel(parsed_args.loglevel)

    main(parsed_args)
