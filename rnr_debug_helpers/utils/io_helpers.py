"""
@author: rchakravarti
@creation-date: 5/9/17
"""
import csv
import gzip
import json
import logging
import os
import sys
from abc import abstractmethod
from collections import OrderedDict
from collections import deque
from configparser import ConfigParser
from warnings import warn

from pkg_resources import resource_filename

from rnr_debug_helpers.utils.predictions import Prediction
from rnr_debug_helpers.utils.queries import LabelledQuery

DEFAULT_ENCODING = 'utf-8'


def initialize_logger(log_level, name):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        # Add std out handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def load_config(config_file_path=resource_filename('config', 'config.ini'), encoding=DEFAULT_ENCODING):
    """
    Loads the config.ini file from the config directory
    :param str or None config_file_path: Uses the default config resource in the package as default if None provided
    :param str or None encoding: Can be used to over ride the default encoding
    :return: ConfigParser initialized with values from the config.ini
    :rtype: ConfigParser
    """
    config = ConfigParser(allow_no_value=True)
    config.read(config_file_path, encoding=encoding)
    return config


def is_gzip_extension(filename):
    return filename.lower().endswith(('.gz', '.gzip'))


def is_cpickle_extension(filename):
    return filename.lower().endswith('.cpickle')


def smart_file_open(filename, mode='r', **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = DEFAULT_ENCODING

    if is_gzip_extension(filename):
        return gzip.open(filename, mode=mode, **kwargs)
    else:
        return open(filename, mode=mode, **kwargs)


def insert_modifier_in_filename(input_file_path, str_to_append=None, file_extension=None):
    """
    generate a path in the same dir as the input file path, but with the "str_to_append" inserted after infile's
        basename (before ".csv, .txt, ...").
    handles .gz extension specially
    For example:
        insert_modifier_in_filename('/home/chakravr/Temp/test-file.txt', 'qnorm')
            --> '/home/chakravr/Temp/test-file.qnorm.txt'
    :param str input_file_path: fully qualified file path that should serve as base
    :param str str_to_append: a string that can be added before the file extension of the input file path
    :param str file_extension: an optional over ride to change the file extension with which the filename is created
    :return: abs filepath with the "str_to_append" added before the file extension of the input file path
    """
    if str_to_append is None and file_extension is None:
        raise ValueError("Either str_to_append or file_extension must be passed, both were None")

    input_filename = os.path.basename(input_file_path)
    gzip_extn = ""
    if is_gzip_extension(input_filename):
        gzip_extn = ".gz"
        input_filename = input_filename[0:-3]

    if file_extension is None:
        file_extension = input_filename[input_filename.rfind(".") + 1:]

    if str_to_append is not None:
        output_file_path = os.path.join(input_file_path, os.path.pardir, "%s.%s.%s" %
                                        (input_filename[0:input_filename.rfind(".")], str_to_append,
                                         file_extension))
    else:
        output_file_path = os.path.join(input_file_path, os.path.pardir, "%s.%s" %
                                        (input_filename[0:input_filename.rfind(".")], file_extension))

    output_file_path += gzip_extn

    return os.path.abspath(output_file_path)


def get_temp_file(original_file):
    return insert_modifier_in_filename(original_file, 'partial')


class PredictionReader:
    __file_has_answer_ids__ = False
    __file_has_confidence_scores__ = False
    __reader__ = None
    __expected_entries_per_line__ = None
    __read_buffer__ = None
    PREDICTION_FILE_DELIMITER = ' '

    def __init__(self, fh, file_has_answer_ids=True, file_has_confidence_scores=False):
        """
        Wraps a csv reader to facilitate reading predictions.  Returns one prediction at a time
        :param FileIO fh: file to read from (expecting output format from run_raas_local_experiment.sh)
        :param bool file_has_answer_ids: optional indicator to say first two columns of file are qid and aid
            respectively
        :param bool file_has_confidence_scores: optional indicator to say the last column of file contains conf score
        """
        self.__reader__ = csv.reader(fh, delimiter=self.PREDICTION_FILE_DELIMITER)
        self.__file_has_answer_ids__ = file_has_answer_ids
        self.__file_has_confidence_scores__ = file_has_confidence_scores
        self.__expected_entries_per_line__ = self._calculate_expected_entries_per_line(file_has_answer_ids,
                                                                                       file_has_confidence_scores)

    def __str__(self):
        return 'PredictionReader(reader=%s, file_has_confidence_scores=%s)' % \
               (self.__reader__, self.__file_has_confidence_scores__)

    def __repr__(self):
        return self.__str__()

    def is_configured_with_confidence_scores(self):
        return self.__file_has_confidence_scores__

    def is_configured_with_answer_ids(self):
        return self.__file_has_answer_ids__

    @staticmethod
    def _calculate_expected_entries_per_line(file_has_answer_ids, file_has_confidence_scores):
        """
        determine expected number of entries on each line of the prediction file
        :param file_has_answer_ids: should we expect two columns in the beginning containing qid and aid
        :param file_has_confidence_scores: should we expect a column at the end with the conf score
        :return: expected number of entries per line
        :rtype: int
        """
        expected_entries_per_line = 1
        if file_has_confidence_scores:
            expected_entries_per_line += 1
        if file_has_answer_ids:
            # if there are answer ids, there must also be question ids...
            expected_entries_per_line += 2

        return expected_entries_per_line

    def get_all_predictions_till_next_query(self):
        """
        Reads all the predictions from the file uptil the next query id boundary.  Throws StopIteration if already
        reached end of file.
        :return: A map from answer id to the predictions for this query
        :rtype: dict(str,Prediction)
        """
        if not self.is_configured_with_answer_ids():
            raise ValueError("Cannot automatically read all answers for a query set unless the prediction file is "
                             "configured with answer id and question id columns")

        preds_for_current_query = OrderedDict()

        # read in the first prediction for this query
        next_prediction_in_file = self.next()
        preds_for_current_query[next_prediction_in_file.aid] = next_prediction_in_file
        prev_qid = next_prediction_in_file.qid

        # continue reading till we see a new query or reach end of file
        while True:
            try:
                next_prediction_in_file = self.next()
                if prev_qid != next_prediction_in_file.qid:
                    # shouldn't have read this pred, put it in buffer and exit loop (we're done)
                    self.__read_buffer__ = next_prediction_in_file
                    break
                elif next_prediction_in_file.aid in preds_for_current_query:
                    raise ValueError("Encountered multiple prediction rows containing answer id <<%s>>"
                                     " for the same qid <<%s>>" % (next_prediction_in_file.aid,
                                                                   next_prediction_in_file.qid))
                else:
                    preds_for_current_query[next_prediction_in_file.aid] = next_prediction_in_file
            except StopIteration:
                # Reached end of file
                break

        return preds_for_current_query

    def next(self):
        """
        Reads the next line from the prediction file and parses it into a prediction object
        :return: parsed prediction
        :rtype: Prediction
        """
        if self.__read_buffer__:
            next_prediction = self.__read_buffer__
            self.__read_buffer__ = None
        else:
            row = next(self.__reader__)
            if len(row) != self.__expected_entries_per_line__:
                raise ValueError("Expected %d entries on each line of prediction file, instead found %d entries: %s" % (
                    self.__expected_entries_per_line__, len(row), row))

            qid = None
            aid = None
            conf_score = None
            if self.__file_has_answer_ids__:
                # first column is
                qid = row.pop(0)
                aid = row.pop(0)
            if self.__file_has_confidence_scores__:
                conf_score = float(row.pop())
            rank_score = float(row[0])

            next_prediction = Prediction(qid=qid, aid=aid, rank_score=rank_score, conf_score=conf_score)

        return next_prediction

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class LabelledQueryStream(object):
    """
    Wraps a reader to facilitate reading lines into a Query.  Returns one query at at time.
    """

    def __init__(self, fh, logger=initialize_logger(logging.INFO, 'LabelledQueryStream')):
        """
        :param File fh: the file containing the ground truth data
        :param logging.Logger or None: an initialized logger or None if you want default logging
        """
        self.logger = logger
        self.query_file = fh
        self._num_questions = 0
        self.logger.debug('Reading ground truth queries from: %s' % self.query_file)

    @abstractmethod
    def next(self):
        """
        Returns the next LabelledQuery in the file
        :return: the next LabelledQuery in the file
        :rtype: LabelledQuery
        """
        raise NotImplementedError('Needs to be over-ridden')

    def __next__(self):
        """
        Duplicate of self.next for Python 3.X compatibility
        """
        return self.next()

    def __iter__(self):
        return self

    def reset(self):
        self.query_file.seek(0)
        self._num_questions = 0

    def __str__(self):
        return 'LabelledQueryStream(fh: %s)' % self.query_file

    def __repr__(self):
        return 'LabelledQueryStream(fh: %s, num quesitons read: %s)' % (self.query_file, self._num_questions)


class RankerRelevanceFileQueryStream(LabelledQueryStream):
    """
    A reader that knows how to parse queries (and corresponding labels) from the `relevance file` that's described
    in the [Training a ranker by using the train.py script](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/training_data.html#script)
    and is the input format for train.py
    """

    def __init__(self, *args, **kwargs):
        super(RankerRelevanceFileQueryStream, self).__init__(*args, **kwargs)

        dialect = csv.excel

        # The following explicit assignments shadow the dialect defaults
        # but are necessary to avoid strange behavior while called by
        # certain unit tests. Please do not delete.
        dialect.doublequote = True
        dialect.quoting = csv.QUOTE_MINIMAL
        dialect.skipinitialspace = True

        self.__reader__ = csv.reader(self.query_file, dialect=dialect)

    def next(self):
        self._num_questions += 1
        try:
            query_text, aids, labels = self._parse_values(next(self.__reader__))
        except StopIteration:
            self.logger.debug('Done iterating over {} lines'.format(self._num_questions))
            raise
        except Exception as ex:
            self.logger.error('Unable to parse values from line %d of file %s due to error: %s' %
                              (self._num_questions, self.query_file, ex))
            raise
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('Read from line %d: %s' % (
                self._num_questions, {'query text': query_text, 'labels': labels, 'doc ids': aids}))

        return LabelledQuery(qid=query_text, labels=labels, ans_ids=aids)

    @staticmethod
    def _parse_values(row):
        if len(row) < 1:
            raise ValueError('Expected line with at least 1 column, but found: <<%s>>' % row)
        qid = row[0]
        aids, labels = list(), list()
        for i in range(1, len(row), 2):
            if len(row) >= (i + 1):
                aids.append(row[i])
                labels.append(int(row[i + 1]))
            else:
                raise ValueError("Even number of columns encountered: %s, expected row in format:"
                                 " <query_text>(, <aid_1>, <label_1>,...,<aid_n>,<label_n>)" % row)
        if len(aids) == 0:
            warn('No labelled answer ids read from row: %s' % row)
        return qid, aids, labels


class RankerFeatureFileStream(RankerRelevanceFileQueryStream):
    """
    A reader that knows how to parse a feature file for the ranker
    """
    __file_has_answer_ids__ = False
    __buffered_row__ = None
    __reader__ = None
    num_features = None
    feature_names = None
    _RAAS_AID_COL_IDX = 1
    _RAAS_QID_COL_IDX = 0
    _RAAS_GT_COL_IDX = -1

    def __init__(self, file_has_answer_ids=False, *args, **kwargs):
        """
        :param fh: the file containing the input in RaaS CSV format (expects header row)
        :param bool file_has_answer_ids: indicates whether or not to expect an answer id in the second col
        """
        super(RankerFeatureFileStream, self).__init__(*args, **kwargs)
        self.__file_has_answer_ids__ = file_has_answer_ids
        self.initialize_feature_names()

    def initialize_feature_names(self):
        self.feature_names = next(self.__reader__)

        # validate expected number of fields
        if self.__file_has_answer_ids__:
            if len(self.feature_names) < 4:
                raise ValueError(
                    "Expected at least 4 feature names in the raas csv header line, found: %s" % self.feature_names)

            # drop the answer id column from the feature names
            self.feature_names.pop(self._RAAS_AID_COL_IDX)
        else:
            if len(self.feature_names) < 3:
                raise ValueError(
                    "Expected at least 3 feature names in the raas csv header line, found: %s" % self.feature_names)

        # drop the qid and ground truth columns from the feature names
        self.feature_names.pop(self._RAAS_QID_COL_IDX)
        self.feature_names.pop(self._RAAS_GT_COL_IDX)
        self.num_features = len(self.feature_names)

    def reset(self):
        super()
        self.__buffered_row__ = None

    def next(self):
        self._num_questions += 1
        if self.__buffered_row__ is not None:
            prev_qid, aid, label, feature_values = self._parse_row_values(self.__buffered_row__)
            self.__buffered_row__ = None
        else:
            prev_qid, aid, label, feature_values = self._parse_row_values(next(self.__reader__))

        labels = [label]
        feature_vectors = [feature_values]
        aids = [aid]

        for row in self.__reader__:
            if row:
                qid, aid, label, feature_values = self._parse_row_values(row)
                if qid != prev_qid:
                    self.__buffered_row__ = row
                    break
                else:
                    labels += [label]
                    feature_vectors += [feature_values]
                    aids += [aid]

        if self.__file_has_answer_ids__:
            return LabelledQuery(qid=prev_qid, feature_vectors=feature_vectors, labels=labels, ans_ids=aids)
        else:
            return LabelledQuery(qid=prev_qid, feature_vectors=feature_vectors, labels=labels)

    def _parse_row_values(self, row):
        try:
            if self.__file_has_answer_ids__:
                if len(row) < 4:
                    raise ValueError('Expected line with at least 4 columns, but found %d: <<%s>>' % (len(row), row))
                aid = row[self._RAAS_AID_COL_IDX]
                feature_values = [float(x) if x else 0.0 for x in
                                  row[(self._RAAS_AID_COL_IDX + 1):self._RAAS_GT_COL_IDX]]
            else:
                if len(row) < 3:
                    raise ValueError('Expected line with at least 3 columns, but found %d: <<%s>>' % (len(row), row))
                feature_values = [float(x) if x else 0.0 for x in
                                  row[(self._RAAS_QID_COL_IDX + 1):self._RAAS_GT_COL_IDX]]
                aid = None

            qid = row[self._RAAS_QID_COL_IDX]
            label = int(row[self._RAAS_GT_COL_IDX])
        except StopIteration:
            self.logger.debug('Done iterating over %d lines' % self._num_questions)
            raise
        except Exception as ex:
            self.logger.error('Unable to parse values from line %d of file %s due to error: %s' %
                              (self._num_questions, self.query_file, ex))
            raise

        return qid, aid, label, feature_values

    def __iter__(self):
        return self


class RnRToolingExportFileQueryStream(LabelledQueryStream):
    """
    A reader that knows how to parse queries (and corresponding labels) from the `export-questions.json` file
    that can be exported from the RnR Tooling
    in the [Training a ranker by using RnR Web UI](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/ranker_tooling.html)
    """
    _GOOD_RATING_THRESHOLD = 3

    def __init__(self, *args, **kwargs):
        super(RnRToolingExportFileQueryStream, self).__init__(*args, **kwargs)

        self.__reader__ = deque(json.load(self.query_file, encoding=DEFAULT_ENCODING))

    def next(self):
        try:
            question = self.__reader__.popleft()
        except IndexError:
            raise StopIteration('Finished processing %d queries' % self._num_questions)

        self._num_questions += 1
        return self._parse_values(question)

    @staticmethod
    def _get_query_text_from_question(question):
        return question['text'].strip()

    def _get_answer_ids_and_labels_from_question(self, question):
        answer_ids = list()
        labels = list()
        for answer_cluster in question['cluster']['answers'].keys():
            for answer in question['cluster']['answers'][answer_cluster]:
                answer_ids.append(answer['id'])
                labels.append(self._map_tooling_star_rating_to_label(answer['ranking']))
        return answer_ids, labels

    def _parse_values(self, question):
        try:
            qtext = self._get_query_text_from_question(question)
            answer_ids, labels = self._get_answer_ids_and_labels_from_question(question)
            parsed_query = LabelledQuery(qid=qtext, labels=labels, ans_ids=answer_ids)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug('Read from question %d: %s' % (self._num_questions, parsed_query))

        except Exception as ex:
            self.logger.error('Unable to parse values from question %d (%s) of file %s due to error: %s' %
                              (self._num_questions, question, self.query_file, ex))
            raise

        return parsed_query

    def __next__(self):
        """
        Duplicate of self.next for Python 3.X compatibility
        """
        return self.next()

    def _map_tooling_star_rating_to_label(self, rating):
        """
        Tooling maps their star ratings to relevance labels for the ranker using this logic:
         {1,2} --> 0, 3 --> 3, 4-->4
        https://github.ibm.com/WatsonTooling/ibmwatson-rr-common/blob/develop/lib/watson/rankers.js#L206

        The reason is that a rating of 1 star and 2 star tends to basically be irrelevant based on the
        guidance in the UI, so they should be marked as such for the ranker.

        :param str rating:
        :return: int
        """
        if int(rating) < self._GOOD_RATING_THRESHOLD:
            return 0
        else:
            return rating


def initialize_query_stream(ground_truth_file, file_format):
    """
    Helper method to initialize the query stream with the appropriate subclass of LabelledQueryStream
    :param File ground_truth_file: file handle to the original file
    :param str file_format: Either 'relevance_file' or 'rnr_tooling_export'. TODO: convert to enum
    :return: An appropriate subclass LabelledQueryStream initialized with the input file handle
    :rtype: LabelledQueryStream
    """
    if file_format == 'relevance_file':
        query_stream = RankerRelevanceFileQueryStream(fh=ground_truth_file)
    elif file_format == 'rnr_tooling_export':
        query_stream = RnRToolingExportFileQueryStream(fh=ground_truth_file)
    else:
        raise NotImplementedError('Unsupported file format {}. Unable to initialize query stream'.format(file_format))
    return query_stream
