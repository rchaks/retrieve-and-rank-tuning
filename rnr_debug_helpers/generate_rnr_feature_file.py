import csv
import json
import logging
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from os import path

from rnr_debug_helpers.utils.rnr_wrappers import RetrieveAndRankProxy
from rnr_debug_helpers.utils.io_helpers import initialize_logger, smart_file_open, RankerRelevanceFileQueryStream, \
    RnRToolingExportFileQueryStream, load_config

LOGGER = initialize_logger(logging.INFO, path.basename(__file__))
_ANS_ID_FIELD = 'id'
_ANS_ID_COL = 0
_QID_COLUMN_NAME = 'question_id'
_GT_COLUMN_NAME = 'relevance_label'


def generate_rnr_features(in_query_stream, outfile, collection_id, cluster_id, num_rows=30, config=load_config()):
    """
    Iterates over a labelled query stream and generates a feature file with the columns:
        <query_num>,<answer_id>,<fea_0>,<fea_1>,...,<fea_n>,<relevance_label>
    :param rnr_debug_helpers.queries.LabelledQueryStream in_query_stream:
    :param File outfile: where the feature file contents will be written to
    :param str collection_id: the RnR solr collection to use for finding search results
    :param str cluster_id: the RnR solr cluster id to use for finding search results
    :param int or None num_rows: The number of search results that will be retrieved for each query. Defaults to 30
        similar to RnR Web UI/Tooling
    :param ConfigParser config: A config loaded with the credentials to use
    """

    rnr_cluster = RetrieveAndRankProxy(solr_cluster_id=cluster_id, config=config)
    writer = csv.writer(outfile)
    # Iterate over queries and generate feature vectors
    stats = defaultdict(int)
    is_first_row = True

    for qid, query in enumerate(in_query_stream):
        labels_for_relevant_answer_ids = _parse_correct_answer_ids_from_query(query)
        _collect_stats(stats, labels_for_relevant_answer_ids)

        LOGGER.debug("Getting feature vectors for query:<<%s>>" % query.get_qid())
        rnr_search_results = rnr_cluster.get_fcselect_features(query_text=query.get_qid(), collection_id=collection_id,
                                                               generate_header=is_first_row,
                                                               num_results_to_return=num_rows)
        if len(rnr_search_results) == 0:
            stats["num_queries_with_zero_rnr_results"] += 1
        else:
            if is_first_row:
                writer.writerow([_QID_COLUMN_NAME] + rnr_search_results.pop(0) + [_GT_COLUMN_NAME])
                is_first_row = False

            stats["num_queries_with_atleast_one_search_result"] += 1
            stats['num_search_results_retrieved'] += len(rnr_search_results)
            num_possible_correct, num_correct_answers_in_search_results = \
                _print_feature_vectors_and_check_for_correct_answers(writer, rnr_search_results, '%d' % (qid + 1),
                                                                     labels_for_relevant_answer_ids)
            if num_possible_correct != num_correct_answers_in_search_results:
                stats['num_queries_where_at_least_correct_answer_didnt_appear_in_rnr'] += 1
            stats["num_correct_in_search_result"] += num_correct_answers_in_search_results

        if stats["num_queries"] % 100 == 0:
            LOGGER.info("Processed %d queries from input file" % stats['num_queries'])
    _average_stats_across_collection(stats)
    LOGGER.info("Finished processing %d queries from input file" % stats['num_queries'])
    return stats


def _parse_correct_answer_ids_from_query(query):
    return {aid: query.get_label(aid) for aid in query.get_answer_ids() if query.get_label(aid) > 0}


def _collect_stats(stats, labels_for_relevant_answer_ids):
    stats["num_queries"] += 1
    stats["num_correct_in_gt_file"] += len(labels_for_relevant_answer_ids)
    if len(labels_for_relevant_answer_ids) == 0:
        stats['num_queries_with_no_correct_answers_specified'] += 1
    for label in labels_for_relevant_answer_ids.values():
        stats["num_occurrences_of_label_%s" % label] += 1


def _print_feature_vectors_and_check_for_correct_answers(writer, rnr_search_results, qid, correct_ans_lookup):
    """
    write the search results to file as a feature vector with the qid and gt labels from the query.
    :param csv.writer writer:
    :param list(list(str)) rnr_search_results:
    :param str qid: the qid to print at the start of each feature vector
    :param dict(str,int) correct_ans_lookup: label lookup for correct answer ids
    :return: num_possible_correct, num_correct_answers_in_search_results
    :rtype: tuple(int,int)
    """
    num_possible_correct = len(correct_ans_lookup)
    num_correct_answers_in_search_results = 0

    for row in rnr_search_results:
        gt_label = 0
        doc_id = row[_ANS_ID_COL].strip()
        if doc_id in correct_ans_lookup:
            gt_label = correct_ans_lookup[doc_id]
            num_correct_answers_in_search_results += 1
        writer.writerow([qid] + row + [gt_label])

    return num_possible_correct, num_correct_answers_in_search_results


def _average_stats_across_collection(stats):
    stats['avg_num_correct_answers_per_query_in_gt_file'] = float(stats["num_correct_in_gt_file"]) / stats[
        'num_queries']
    stats['avg_num_correct_answers_per_query_in_rnr_results_default'] = \
        float(stats["num_correct_in_search_result"]) / stats['num_queries_with_atleast_one_search_result']
    stats['avg_num_search_results_retrieved_per_query'] = stats['num_search_results_retrieved'] / float(
        stats["num_queries_with_atleast_one_search_result"])


def main(command_line_args):
    LOGGER.info('Start Script')
    LOGGER.info("Iterating through labelled queries from: %s" % command_line_args.input_file)

    with smart_file_open(command_line_args.input_file, mode='r') as infile:
        if command_line_args.infile_format == 'relevance_file':
            labelled_queries = RankerRelevanceFileQueryStream(infile)
        elif command_line_args.infile_format == 'rnr_tooling_export':
            labelled_queries = RnRToolingExportFileQueryStream(infile)
        else:
            raise NotImplementedError('Unsupported file format: %s' % command_line_args.infile_format)

        if command_line_args.output_file is not None:
            LOGGER.info('Writing feature file to: %s' % command_line_args.output_file)
            with smart_file_open(command_line_args.output_file, mode='w') as outfile:
                stats = generate_rnr_features(labelled_queries, outfile, command_line_args.collection_id,
                                              command_line_args.solr_cluster_id)
        else:
            LOGGER.info('Feature file will be printed to stdout')
            stats = generate_rnr_features(labelled_queries, sys.stdout, command_line_args.collection_id,
                                          command_line_args.solr_cluster_id)
    json.dump(stats, sys.stdout, sort_keys=True, indent=4)
    LOGGER.info("Feature file generation complete")


if __name__ == '__main__':
    parser = ArgumentParser(prog="python %s)" % path.basename(__file__),
                            description="Helper script to generate feature files using the RnR /fcselect API",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_file', dest='input_file', help="file path to read labelled queries from",
                        required=True)
    parser.add_argument('-o', '--output_file', dest='output_file', required=False, default=None,
                        help="optionally provide a file path to output the feature file to. If one isn't provided, we"
                             " ouput to std out")
    parser.add_argument('-c', '--collection_id', dest='collection_id', required=True,
                        help="The collection id that will be used to generate features ")
    parser.add_argument('-s', '--solr_cluster_id', dest='solr_cluster_id', required=True,
                        help="The solr (RnR) cluster id that will be used to generate features ")
    parser.add_argument('-f', '--file_format', dest='infile_format', choices=['relevance_file', 'rnr_tooling_export'],
                        default='relevance_file',
                        help="indicate whether the input file is in the `relevance file` format that's described in "
                             "the [Training a ranker by using the train.py script](https://www.ibm.com/watson/devel"
                             "opercloud/doc/retrieve-rank/training_data.html#script) or the `export-questions.json` "
                             "file that can be exported from the RnR Tooling in the [Training a ranker by using the "
                             "train.py script](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/ranker_"
                             "tooling.html)")
    parser.add_argument('-d', '--debug', help="Print lots of debugging statements", action="store_const",
                        dest="log_level", const=logging.DEBUG, default=logging.INFO)
    args = parser.parse_args()
    LOGGER = initialize_logger(args.log_level, path.basename(__file__))
    # get on with it
    main(args)