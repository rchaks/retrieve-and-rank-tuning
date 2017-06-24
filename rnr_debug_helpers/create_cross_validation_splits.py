import argparse
import csv
import logging
from os import makedirs, path

from rnr_debug_helpers.utils.io_helpers import initialize_query_stream, smart_file_open, initialize_logger

TRAIN_RELEVANCE_FILENAME = "train.relevance_file.csv"
VALIDATION_RELEVANCE_FILENAME = 'validation.relevance_file.csv'
LOGGER = initialize_logger(logging.INFO, path.basename(__file__))


def k_fold_cross_validation(X, K, randomise=False):
    """
    Taken from http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.

    :param iterable X: set of data points to split into folds
    :param int K: number of folds to create splits for
    :param bool randomise: whether or not incoming list should be randomized
    :return: a train and validation split for each fold.  Implemented as generator, so each call yields a new
        split for a new fold up to k
    :rtype: tuple(list, list)
    """
    if randomise:
        from random import shuffle
        X = list(X)
        shuffle(X)

    for k in range(K):
        training = set([x for i, x in enumerate(X) if i % K != k])
        validation = set([x for i, x in enumerate(X) if i % K == k])
        yield training, validation


def setup_output_writers(parent_dir, fold_number):
    """
    Create an output directory for the fold under the provided parent dir
    :param str parent_dir: file path to the output dir
    :param int fold_number: fold number to use in directory name
    :return: writer for <outdir.name>/fold<fold_number>/train.csv and <outdir.name>/fold<fold_number>/validation.csv
    :rtype: tuple(csv.writer,csv.writer)
    """
    output_dir = path.join(parent_dir, "Fold%d" % fold_number)
    if not path.isdir(output_dir):
        LOGGER.debug("Creating output for fold %d at the location: %s" % (fold_number, output_dir))
        makedirs(output_dir)
    else:
        LOGGER.warn("Path <<%s>> already exists, files may be overwritten" % output_dir)

    train_writer = csv.writer(smart_file_open(path.join(output_dir, TRAIN_RELEVANCE_FILENAME), 'w'),
                              dialect=csv.excel, delimiter=',')
    validation_writer = csv.writer(smart_file_open(path.join(output_dir, VALIDATION_RELEVANCE_FILENAME), 'w'),
                                   dialect=csv.excel, delimiter=',')

    return train_writer, validation_writer


def setup_train_and_test_writer(output_dir):
    """
    Create an output directory for the fold under the provided parent dir
    :param str output_dir: file path to the output dir
    :return: writer for <outdir.name>/train.csv and <outdir.name>/validation.csv
    :rtype: tuple(csv.writer,csv.writer)
    """
    if not path.isdir(output_dir):
        makedirs(output_dir)
    else:
        LOGGER.warn("Path <<%s>> already exists, files may be overwritten" % output_dir)

    train_writer = csv.writer(smart_file_open(path.join(output_dir, TRAIN_RELEVANCE_FILENAME), 'w'),
                              dialect=csv.excel, delimiter=',')
    validation_writer = csv.writer(smart_file_open(path.join(output_dir, VALIDATION_RELEVANCE_FILENAME), 'w'),
                                   dialect=csv.excel, delimiter=',')

    return train_writer, validation_writer


def main(args):
    LOGGER.info('Generating splits from original file: %s' % args.infile)
    LOGGER.info('Splits will be created in directory: %s' % args.output_path)
    with smart_file_open(args.infile) as infile:
        split_files_into_k_cv_folds(initialize_query_stream(infile, args.infile_format),
                                    args.output_path, k=args.k)


def split_files_into_k_cv_folds(labelled_query_stream, outdir, k):
    unique_question_numbers = [j for j, _ in enumerate(labelled_query_stream)]

    i = 1
    for train_qids, validation_qids in k_fold_cross_validation(unique_question_numbers, K=k, randomise=True):
        LOGGER.info("Creating train and test splits for fold %d" % i)
        # reset the query stream so that we can re-read from the start of the file
        labelled_query_stream.reset()

        num_questions_in_train_fold, num_test_instances = 0, 0
        train_writer, test_writer = setup_output_writers(outdir, i)

        for query_number, query in enumerate(labelled_query_stream):
            if query_number in train_qids:
                train_writer.writerow(query.to_csv_row_for_relevance_file())
                num_questions_in_train_fold += 1
            else:
                test_writer.writerow(query.to_csv_row_for_relevance_file())
                num_test_instances += 1

        LOGGER.info("Wrote %d train instances and %d test instances to <<%s>>" % (
            num_questions_in_train_fold, num_test_instances, path.join(outdir, "Fold%d" % i)))
        i += 1


def split_file_into_train_and_test(labelled_query_stream, outdir, train_percentage=0.8):
    unique_question_numbers = [j for j, _ in enumerate(labelled_query_stream)]
    num_train_qids = int(len(unique_question_numbers) * train_percentage)
    train_qids = unique_question_numbers[0:num_train_qids]
    validation_qids = unique_question_numbers[num_train_qids:]

    LOGGER.info("Creating train split with %d queries (%.1f %%) and test split with %d queries" %
                (len(train_qids), train_percentage * 100, len(validation_qids)))
    # reset the query stream so that we can re-read from the start of the file
    labelled_query_stream.reset()

    num_questions_in_train_fold, num_test_instances = 0, 0
    train_writer, test_writer = setup_train_and_test_writer(outdir)

    for query_number, query in enumerate(labelled_query_stream):
        if query_number in train_qids:
            train_writer.writerow(query.to_csv_row_for_relevance_file())
            num_questions_in_train_fold += 1
        else:
            test_writer.writerow(query.to_csv_row_for_relevance_file())
            num_test_instances += 1

    LOGGER.info("Wrote %d train instances and %d test instances to <<%s>>" % (
        num_questions_in_train_fold, num_test_instances, outdir))


if __name__ == '__main__':
    # Get cmd line args
    parser = argparse.ArgumentParser(description='Helper script to split ranker relevance csv file or feature files'
                                                 ' into Cross Validation train & test folds based on question ids')
    parser.add_argument('-i', '--inFile', dest='infile', required=True,
                        help="input file path to split by question id into k folds")
    parser.add_argument('-o', '--outDir', dest='output_path', required=True,
                        help="output directory path where sub-directories containing train and test files will be "
                             "created for each fold")
    parser.add_argument('-k', '--numFolds', dest='k', type=int, default=5, help="Number of CV folds to generate")
    parser.add_argument('-f', '--file_format', dest='infile_format', choices=['relevance_file', 'rnr_tooling_export'],
                        default='relevance_file',
                        help="indicate whether the input file is in the `relevance file` format that's described in "
                             "the [Training a ranker by using the train.py script](https://www.ibm.com/watson/devel"
                             "opercloud/doc/retrieve-rank/training_data.html#script) or the `export-questions.json` "
                             "file that can be exported from the RnR Tooling in the [Training a ranker by using the "
                             "train.py script](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/ranker_"
                             "tooling.html)")
    parser.add_argument('-d', '--debug', help="Print lots of debugging statements", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    cl_args = parser.parse_args()

    # set log level
    LOGGER = initialize_logger(cl_args.loglevel, path.basename(__file__))

    main(cl_args)
