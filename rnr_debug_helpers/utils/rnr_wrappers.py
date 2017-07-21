"""
Just holds some constants
@author: rchakravarti
@creation-date: 9/20/16
"""
import csv
import json
import logging
import sys
import tempfile
import urllib.parse
from collections import defaultdict
from datetime import timedelta
from os import path
from pprint import pprint
from shutil import move
from time import sleep
from warnings import warn

import requests
import requests.exceptions
from watson_developer_cloud import RetrieveAndRankV1, WatsonException

from rnr_debug_helpers.utils.io_helpers import load_config, initialize_logger, smart_file_open, get_temp_file, \
    insert_modifier_in_filename, is_gzip_extension, RankerFeatureFileStream
from rnr_debug_helpers.utils.predictions import Prediction

# Lucene query syntax treats these characters specially...so they need to be escaped
CHARS_THAT_HAVE_TO_BE_ESCAPED = ["\\", "+", "-", "&&", "||", "!", "(", ")", "{", "}", "[", "]", "^", '"', "~", "*",
                                 "?", ":", "/"]
SOLR_KEYWORDS_THAT_HAVE_TO_BE_LOWER_CASED = ["AND", "OR", "NOT"]
_QID_COL_INDEX = 0
_ANS_ID_COL_INDEX = 1
_LABEL_COL_INDEX = -1
_MAX_UPLOAD_SIZE = 314572800  # 300 MB


def escape_special_tokens_for_solr_syntax(text):
    modified_text = ""
    for word in text.split():
        if word in SOLR_KEYWORDS_THAT_HAVE_TO_BE_LOWER_CASED:
            word = '%s' % word.lower()
        modified_text += " %s" % word

    for char in CHARS_THAT_HAVE_TO_BE_ESCAPED:
        modified_text = modified_text.replace(char, "\%s" % char)

    return modified_text


def get_rnr_credentials(config):
    url = config.get('RetrieveAndRank', 'url', fallback=RetrieveAndRankV1.default_url)
    user_id = config.get('RetrieveAndRank', 'user')
    password = config.get('RetrieveAndRank', 'password')
    return url, user_id, password


class AbstractBluemixProxy(object):
    """
        My convenience wrapper for the Bluemix API calls as well as helper methods
    """
    _MAX_RETRIES = 2
    _SECONDS_BETWEEN_STATUS_POLLING_REQUESTS = 60

    def __init__(self, config=load_config(), logger=initialize_logger(logging.INFO, 'BluemixServiceProxy')):
        """

        :param ConfigParser config: for access to credentials
        :param logging.Logger logger: for logging
        """
        self.logger = logger
        self.config = config
        self.bluemix_url, self.bluemix_user, self.bluemix_password = get_rnr_credentials(config)
        self.bluemix_connection = RetrieveAndRankV1(url=self.bluemix_url, username=self.bluemix_user,
                                                    password=self.bluemix_password)

    def _handle_exception(self, ex, num_attempts, message):
        if num_attempts < self._MAX_RETRIES:
            self.logger.warn("%s. Retrying." % message)
            # re-initiating the connection sometimes does the trick
            self.bluemix_connection = RetrieveAndRankV1(url=self.bluemix_url, username=self.bluemix_user,
                                                        password=self.bluemix_password)
        else:
            raise RuntimeError("%s.  Giving up on re-attempts, last failure reason: %s" % (message, ex)) from ex


class RetrieveAndRankProxy(AbstractBluemixProxy):
    """
    My convenience wrapper for the RetrieveAndRank API calls as well as helper methods to pre-process/post-process
        data going into and out of those API calls.
    """
    DOC_ID_FIELD_NAME = 'id'

    def __init__(self, solr_cluster_id=None, cluster_name='TestCluster', cluster_size='7', *args, **kwargs):
        """
        Initialize the connection to a Bluemix RnR cluster

        :param str or None solr_cluster_id: Either provide a previously created cluster id or pass None if you want a
            new one to be created
        :param str cluster_name: If a new cluster has to be created, this is the name the cluster will receive
        :param int or str cluster_size: Either an int representing the cluster size (see bluemix doc) or the str
            representation of the same
        """
        super(RetrieveAndRankProxy, self).__init__(*args, **kwargs)

        if solr_cluster_id is None:
            self.logger.info("Creating a new cluster with name %s and size %s" % (cluster_name, cluster_size))
            self.solr_cluster_id = self._create_cluster(cluster_name=cluster_name, cluster_size=str(cluster_size))
        else:
            self.logger.info("Using previously created solr cluster id: %s" % solr_cluster_id)
            self.solr_cluster_id = solr_cluster_id
            json.dump(self.bluemix_connection.get_solr_cluster_status(solr_cluster_id), sys.stdout, sort_keys=True,
                      indent=4)

    def get_pysolr_client(self, collection_id):
        return self.bluemix_connection.get_pysolr_client(self.solr_cluster_id, collection_id)

    def _get_runtime_predictions(self, question_number, query_text, collection_id, num_results_to_return,
                                 ranker_id=None):
        query_parameters = {'q': escape_special_tokens_for_solr_syntax(query_text).encode('utf-8'),
                            'wt': 'json',
                            'rows': num_results_to_return,
                            'fl': 'score,ranker.confidence,%s' % self.DOC_ID_FIELD_NAME}
        if ranker_id is not None:
            query_parameters['ranker_id'] = ranker_id
        response = None
        try:
            response = requests.get("%s/v1/solr_clusters/%s/solr/%s/fcselect" % (self.bluemix_url, self.solr_cluster_id,
                                                                                 collection_id),
                                    params=urllib.parse.urlencode(query_parameters),
                                    auth=(self.bluemix_user, self.bluemix_password))
            response.raise_for_status()
            predictions = self._parse_response_content_for_predictions(question_number, response)
        except Exception as ex:
            self.logger.error("Error completing /fcselect request for query <<%s>>: %s" % (query_text, str(ex)))
            if response is not None:
                pprint(vars(response))
            raise
        return predictions, response.elapsed

    def generate_fcselect_prediction_scores(self, test_questions, prediction_file_location, collection_id, num_rows=10,
                                            ranker_id=None):
        """
        Generates runtime requests using the data from the input test file, submits them to the ranker associated with
            the input ranker id and writes returned predictions to the specified output path.  The predictions are in
            the same sequence as the feature vectors in the test file. However, since RaaS only returns top 10 ranked
            documents the remaining document scores are defaulted to -1 (with confidence 0)

        :param LabelledQueryStream test_questions: a csv containing data to use for the requests (specifically
            only care about the question_text)
        :param str prediction_file_location: valid path for the prediction file to be created (over writes existing)
        :param str collection_id: the collection id at which the queries will be pointed to in the cluster
        :param int or None num_rows: the number of predictions to write to the prediction file. Defaults to 10
        :param str or None ranker_id: If available, a ranker id that should be used to re-rank the results
        """
        self.logger.info(
            "Sending runtime requests from <<%s>> to collection: <<%s>> (predictions will be written to: <<%s>>)" % (
                test_questions, collection_id, prediction_file_location))

        if ranker_id is not None:
            self.logger.info('Answers will be re-ranked using the ranker <%s>' % ranker_id)

        temp_file = get_temp_file(prediction_file_location)
        stats = defaultdict(float)
        stats['response_time'] = timedelta(seconds=0)
        with smart_file_open(temp_file, 'w') as prediction_outfile:
            writer = csv.writer(prediction_outfile, delimiter=' ')
            for query in test_questions:
                stats['num_questions'] += 1
                self.logger.debug("Generate predictions for query <<%s>>" % query.get_qid())
                predictions, response_time = self._get_runtime_predictions(stats['num_questions'],
                                                                           query_text=query.get_qid(),
                                                                           collection_id=collection_id,
                                                                           num_results_to_return=num_rows,
                                                                           ranker_id=ranker_id)
                stats['response_time'] += response_time
                if predictions:
                    stats['num_results_returned'] += len(predictions)
                    self._write_results_to_file(predictions, writer)
                else:
                    stats['num_queries_which_doesnt_have_any_results'] += 1
                if self.logger.isEnabledFor(logging.DEBUG) or stats['num_questions'] % 500 == 0:
                    self.logger.info('Generated predictions for %d queries' % stats['num_questions'])

            if stats['num_questions'] < 1:
                raise ValueError("No test instances found in the file")
            stats['avg_response_time'] = stats['response_time'] / stats['num_questions']
        move(temp_file, prediction_file_location)

        self.logger.info("Completed getting runtime predictions for %d questions" % stats['num_questions'])
        return stats

    @staticmethod
    def _write_results_to_file(results, writer):
        for prediction in results:
            if prediction.conf_score is None:
                writer.writerow(['%d' % prediction.qid, '%s' % prediction.aid, '%s' % prediction.rank_score])
            else:
                writer.writerow(['%d' % prediction.qid, '%s' % prediction.aid, '%s' % prediction.rank_score,
                                 '%s' % prediction.conf_score])

    def get_fcselect_features(self, query_text, collection_id, generate_header=False, num_results_to_return=10):
        """
        Helper method issues a GET request to RnR to generate the feature matrix corresponding the retrieved results for
            this query
        :param str query_text: query string (already formatted in solr style syntax)
        :param str collection_id: the name of the collection from the cluster to use (must be previously setup)
        :param bool generate_header: whether or not the returned feature matrix should contain a header row
        :param int num_results_to_return: number of rows to query for in the result set
        :return: list of csv rows representing feature vectors (first column is the answer id) corresponding to the
            retrieved set of documents for this query
        :rtype: list(list(str))
        """
        query_parameters = {'q': escape_special_tokens_for_solr_syntax(query_text).encode('utf-8'),
                            'generateHeader': str(generate_header).lower(),
                            'returnRSInput': 'true',
                            'wt': 'json',
                            'rows': num_results_to_return,
                            'fl': self.DOC_ID_FIELD_NAME}

        response = None
        try:
            response = requests.get("%s/v1/solr_clusters/%s/solr/%s/fcselect" % (self.bluemix_url, self.solr_cluster_id,
                                                                                 collection_id),
                                    params=urllib.parse.urlencode(query_parameters),
                                    auth=(self.bluemix_user, self.bluemix_password))
            response.raise_for_status()
            feature_matrix_from_response = self._parse_response_content_for_features(response)
        except Exception as ex:
            self.logger.error("Error completing /fcselect request for query <<%s>>: %s" % (query_text, str(ex)))
            if response is not None:
                pprint(vars(response))
            raise

        return feature_matrix_from_response

    def _parse_response_content_for_predictions(self, question_number, response):
        """
        Parses the json representation of the docs from the HTTP response and returns it as list predictions
            with scores (and confidences if they exist)
        :param str question_number: used as qid
        :param requests.Response response:
        :return: list of feature vectors
        :rtype: list(list(str))
        """
        response_content = response.json()
        results = []
        if response_content['response']['numFound'] > 0:
            for doc in response_content['response']['docs']:
                if "ranker.confidence" in doc:
                    conf_score = doc["ranker.confidence"]
                else:
                    conf_score = None
                results.append(Prediction(qid=question_number, aid=doc[self.DOC_ID_FIELD_NAME], rank_score=doc['score'],
                                          conf_score=conf_score))
        else:
            self.logger.warn('Empty response: %s' % vars(response))
        return results

    @staticmethod
    def _parse_response_content_for_features(response):
        """
        Parses the json representation of the feature matrix from the HTTP response and returns it as list feature
            vectors
        :param requests.Response response:
        :return: list of feature vectors
        :rtype: list(list(str))
        """
        response_content = response.json()
        if 'RSInput' in response_content:
            return [[str(val) for val in line.split(",")] for line in response_content['RSInput'].split('\n') if line]
        else:
            warn('Empty response: %s' % vars(response))
            return []

    def setup_cluster_and_collection(self, config_id, collection_id, config_zip=None):
        """
        If the provided config and collection ids are not already present in the cluster, creates new ones.
        Requires the config zip in that case
        :param str config_id:
        :param str collection_id:
        :param str or None config_zip:
        :return:
        """
        self.wait_for_cluster_to_become_ready()

        if not self.config_previously_uploaded(config_id):
            self.upload_solr_config(config_id, config_zip)

            if self.collection_previously_created(collection_id):
                raise ValueError("A new config was created, but the provided collection id already exists, so we can't"
                                 " create a new collection to reflect the new config.  Delete this collection first")

        if not self.collection_previously_created(collection_id):
            self.create_collection(collection_id, config_id)

        # Test collection
        self.logger.info("Collection: <<%s>> in cluster: <<%s>> (with config: <<%s>>) setup with %d documents" %
                         (collection_id, self.solr_cluster_id, config_id,
                          self.get_num_docs_in_collection(collection_id)))

    def wait_for_cluster_to_become_ready(self):
        state = self.bluemix_connection.get_solr_cluster_status(solr_cluster_id=self.solr_cluster_id)
        if state[u'solr_cluster_status'] == u'NOT_AVAILABLE':
            self.logger.info("Waiting for cluster <<%s>> to become available" % self.solr_cluster_id)
            while state[u'solr_cluster_status'] == u'NOT_AVAILABLE':
                state = self.bluemix_connection.get_solr_cluster_status(solr_cluster_id=self.solr_cluster_id)
                sleep(self._SECONDS_BETWEEN_STATUS_POLLING_REQUESTS)
        self.logger.info("Solr cluster %s is available for use" % self.solr_cluster_id)
        json.dump(state, sys.stdout, sort_keys=True, indent=4)

    def _create_cluster(self, cluster_name, cluster_size):
        self.logger.info("Submitting request to create a cluster")
        created_cluster = self.bluemix_connection.create_solr_cluster(cluster_name=cluster_name,
                                                                      cluster_size=cluster_size)
        json.dump(created_cluster, sys.stdout, sort_keys=True, indent=4)
        return created_cluster[u'solr_cluster_id']

    def config_previously_uploaded(self, config_id):
        """
        :param str config_id: identifier for config
        """
        previously_uploaded_configs = self.bluemix_connection.list_configs(solr_cluster_id=self.solr_cluster_id)
        for configs in previously_uploaded_configs.values():
            if config_id in configs:
                return True
        return False

    def get_available_configs(self):
        return self.bluemix_connection.list_configs(solr_cluster_id=self.solr_cluster_id).values()

    def get_previously_created_collections(self):
        response = self.bluemix_connection.list_collections(self.solr_cluster_id)
        return response['collections']

    def delete_config(self, config_id):
        if self.config_previously_uploaded(config_id):
            self.logger.info('Sumitting request to delete config: %s' % config_id)
        else:
            raise ValueError('Config id <%s> doesn\'t exist, available configs include: %s' %
                             (config_id, self.get_available_configs()))

    def delete_collection(self, collection_id):
        if self.collection_previously_created(collection_id):
            self.logger.info('Submitting request to delete collection: %s' % collection_id)
            self.bluemix_connection.delete_collection(self.solr_cluster_id, collection_name=collection_id)
        else:
            raise ValueError('Collection <%s> does not exist, available collections include: %s' %
                             (collection_id, self.get_previously_created_collections()))

    def collection_previously_created(self, collection_id):
        response = self.bluemix_connection.list_collections(self.solr_cluster_id)
        return collection_id in response['collections']

    def upload_solr_config(self, config_id, config_zip_filepath):
        if not path.isfile(config_zip_filepath):
            raise ValueError("Missing solr config zip file: %s" % config_zip_filepath)

        with open(config_zip_filepath, 'rb') as zipped_config:
            self.logger.info("Uploading solr configurations")
            config_status = self.bluemix_connection.create_config(self.solr_cluster_id, config_id, zipped_config)
            json.dump(config_status, sys.stdout, sort_keys=True, indent=4)

    def create_collection(self, collection_id, config_id):
        self.logger.info("Creating a collection: %s" % collection_id)
        collection = self.bluemix_connection.create_collection(self.solr_cluster_id, collection_id, config_id)
        json.dump(collection, sys.stdout, sort_keys=True, indent=4)

    def get_num_docs_in_collection(self, collection_id):
        pysolr_client = self.bluemix_connection.get_pysolr_client(self.solr_cluster_id, collection_id)
        results = pysolr_client.search('*:*')
        return results.hits

    def upload_documents_to_collection(self, collection_id, corpus_file, content_type='application/xml'):
        with smart_file_open(corpus_file) as solr_formated_doc:
            response = requests.post(
                "%s/v1/solr_clusters/%s/solr/%s/update" % (self.bluemix_url, self.solr_cluster_id, collection_id),
                data=solr_formated_doc,
                headers={'X-Requested-With': 'Python requests', 'Content-type': content_type,
                         'x-global-transaction-id': 'Rishavs-Pubmed-Upload'},
                params={"commit": "true"},
                auth=(self.bluemix_user, self.bluemix_password))
            if response.status_code != 200:
                pprint(vars(response))
                response.raise_for_status()
            else:
                self.logger.info("Successful doc upload")


class RankerProxy(AbstractBluemixProxy):
    """
    My convenience wrapper for the Ranker specific API calls as well as helper methods to pre-process/post-process
        data going into and out of those API calls.
    """

    def generate_ranker_predictions(self, ranker_id, test_file_location, prediction_file_location,
                                    file_has_answer_ids=True):
        """
        Generates runtime requests using the data from the input test file, submits them to the ranker associated with
            the input ranker id and writes returned predictions to the specified output path.  The predictions are in
            the same sequence as the feature vectors in the test file. However, since RaaS only returns top 10 ranked
            documents the remaining document scores are defaulted to -1 (with confidence 0)

        :param str ranker_id: id for the associated ranker in bluemix
        :param str test_file_location: a csv containing data to use for the requests (question_id, feature_1,
            feature_2,..., label)
        :param str prediction_file_location: valid path for the prediction file to be created (over writes existing)
        :param bool file_has_answer_ids: a flag to indicate whether or not the file has an answer id column (if not,
            one will be mocked)
        """
        self.logger.info(
            "Sending runtime requests from <<%s>> to ranker id: <<%s>> (predictions will be written to: <<%s>>)" % (
                test_file_location, ranker_id, prediction_file_location))

        with smart_file_open(test_file_location) as test_file:
            temp_file = get_temp_file(prediction_file_location)
            stats = defaultdict(float)
            with smart_file_open(temp_file, 'w') as prediction_outfile:
                query_stream = RankerFeatureFileStream(fh=test_file, file_has_answer_ids=file_has_answer_ids)

                for query in query_stream:
                    stats['num_questions'] += 1
                    self.logger.debug("Generate predictions for qid <<%s>>" % query.get_qid())
                    ranked_candidate_answers = self._call_runtime(ranker_id, query, query_stream.feature_names)
                    num_answers_written = self._write_ranker_preds_to_prediction_file(query.get_qid(),
                                                                                      ranked_candidate_answers,
                                                                                      prediction_outfile)
                    if num_answers_written != query.get_answer_count():
                        raise ValueError(
                            "Error getting ranked answers for qid %s.  Expected %d answers, but only got %d: %s" %
                            (query.get_qid(), query.get_answer_count(), num_answers_written,
                             ranked_candidate_answers))
                    sleep(0.001)

                if stats['num_questions'] < 1:
                    raise ValueError("No test instances found in the file")
            move(temp_file, prediction_file_location)

            self.logger.info("Completed getting runtime predictions for %d questions" % stats['num_questions'])

    def _call_runtime(self, ranker_id, query, feature_names):
        """
        Helper method for a single runtime request to the specified ranker id for the candidate answers and question id
            provided as input.
        :param str ranker_id: id associated with the ranker to submit the runtime requests to.
        :param common.query.Query query: query for which rank needs to be called
        :param list(str) feature_names: feature names to use
        :return: list of candidate answers some of them have no rank score (these weren't returned by the service)
        :rtype: list(CandidateAnswer)
        """

        answer_file_headers = ['aid'] + feature_names

        with tempfile.SpooledTemporaryFile(mode='w') as file_to_send_with_request:
            writer = csv.writer(file_to_send_with_request)
            writer.writerow(answer_file_headers)

            for aid in query.get_answer_ids():
                outrow = [aid] + query.get_feature_vector(aid)
                writer.writerow(outrow)

            file_to_send_with_request.flush()
            file_to_send_with_request.seek(0)

            num_attempts_for_this_qid = 0
            response = None
            while True:
                try:
                    num_attempts_for_this_qid += 1
                    response = self.bluemix_connection.rank(ranker_id=ranker_id, answer_data=file_to_send_with_request,
                                                            top_answers=query.get_answer_count())
                    break
                except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as ex:
                    self._handle_exception(ex, num_attempts_for_this_qid,
                                           "Attempt #%d for qid: %s failed." %
                                           (num_attempts_for_this_qid, query.get_qid()))
                except WatsonException as ex:
                    self._handle_exception(ex, num_attempts_for_this_qid,
                                           "Attempt #%d for qid: %s failed." %
                                           (num_attempts_for_this_qid, query.get_qid()))

            if self.logger.isEnabledFor(logging.DEBUG):
                print(json.dumps(response, indent=2))

            self.logger.debug("Runtime request processed <<%d>> candidates for qid: <<%s>>" % (
                query.get_answer_count(), query.get_qid()))

            return response

    @staticmethod
    def _write_ranker_preds_to_prediction_file(qid, response_contents, outfile):
        num_answers = 0
        for ranked_answer in response_contents['answers']:
            num_answers += 1
            outfile.write("%s %s %.4f %.4f\n" %
                          (qid, ranked_answer['answer_id'], ranked_answer['score'], ranked_answer['confidence']))
        return num_answers

    def ensure_feature_file_size_is_ok_for_create_ranker_file_upload(self, train_file_location):
        self.logger.info('Checking file size before making train call for %s' % train_file_location)
        file_size = self._get_file_size(train_file_location)

        if file_size < 1:
            raise ValueError("Looks like train file is empty (%s bytes)??" % file_size)
        elif file_size > _MAX_UPLOAD_SIZE:
            self.logger.warn("File size (%s bytes) is greater than allowable limit (%d bytes)" %
                             (file_size, _MAX_UPLOAD_SIZE))
            self.logger.warn("Generating a shrunk version of the file for upload")
            train_file_location = self.shrink_train_file_to_within_upload_size_limit(train_file_location)
        else:
            self.logger.info("File size looks ok: %d bytes" % file_size)
        return train_file_location

    @staticmethod
    def _get_file_size(file_location):
        # TODO: Add gzip compatibility
        if is_gzip_extension(file_location):
            raise NotImplementedError('Support gzip format')
        else:
            file_size = path.getsize(file_location)
        return file_size

    def shrink_train_file_to_within_upload_size_limit(self, train_file_location):
        file_size_after_spart_format, sparse_csv_format_file = self._generate_sparse_format_file(train_file_location)
        # check if we need to do more
        if file_size_after_spart_format > _MAX_UPLOAD_SIZE:
            # TODO: implement subsampler
            raise ValueError('The feature file is too large (%s) for use with the RnR service (which has a limit of '
                             '%s) even after converting to sparse csv format.  Try randomly subsampling some of the '
                             'incorrect answers per query' % (file_size_after_spart_format, _MAX_UPLOAD_SIZE))

        self.logger.info("Shrunk the file (%s) sufficiently down to %d by using sparse csv format" %
                         (sparse_csv_format_file, file_size_after_spart_format))

        return sparse_csv_format_file

    def _generate_sparse_format_file(self, feature_file):
        sparse_file = insert_modifier_in_filename(feature_file, 'sparse_format')
        if path.isfile(sparse_file):
            self.logger.info("Re-using previously generated sparse format file: %s" % sparse_file)
        else:
            self.logger.info('Generating a sparse version of the feature file (zeros replaced with empty columns '
                             'which the ranker knows how to deal with)')
            temp_file = get_temp_file(sparse_file)
            with smart_file_open(temp_file, 'w') as outfile:
                writer = csv.writer(outfile)
                with smart_file_open(feature_file) as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        writer.writerow(row[:1] + row[2:])
            move(temp_file, sparse_file)
        self.logger.info('Done generating file: %s' % sparse_file)

        return self._get_file_size(sparse_file), sparse_file

    def _drop_answer_id_col_from_feature_file(self, train_file_location):
        file_without_aid = insert_modifier_in_filename(train_file_location, 'no_aid')
        if path.isfile(file_without_aid):
            self.logger.info('Found a previously generated version of the training file without answer id column, '
                             're-using it: %s' % file_without_aid)
        else:
            self.logger.info('Generating a version of the feature file without answer id (which is what ranker'
                             ' training expects')
            temp_file = get_temp_file(file_without_aid)
            with smart_file_open(temp_file, 'w') as outfile:
                writer = csv.writer(outfile)
                with smart_file_open(train_file_location) as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        writer.writerow(row[:1] + row[2:])
            move(temp_file, file_without_aid)
            self.logger.info('Done generating file: %s' % file_without_aid)
        return file_without_aid

    def train_ranker(self, train_file_location, is_enabled_make_space=False, ranker_name='RANKER-EXPERIMENT',
                     train_file_has_answer_id=False):
        """
        Method submits POST request to create a new ranker using the input training file. Then polls the service
            waiting for training to complete. Raises exception if ranker training fails.

        :param str train_file_location: filepath to the training file in csv format (qid,feature1,feature2...,label)
        :param bool is_enabled_make_space: boolean which decides if pre-existing rankers can be deleted to make space
            space for a new ranker.
        :param str ranker_name: shows up with this name in bluemix
        :param bool train_file_has_answer_id: a boolean that means we need to drop the first column from the feature
            file before calling the ranker
        :return: ranker id that can be used to access to the ranker in bluemix
        :rtype: str
        """
        self.logger.info("Submitting request to create a new ranker trained with file %s" % train_file_location)
        if train_file_has_answer_id:
            train_file_location = self._drop_answer_id_col_from_feature_file(train_file_location)
        train_file_location = self.ensure_feature_file_size_is_ok_for_create_ranker_file_upload(train_file_location)
        with smart_file_open(train_file_location) as train_file:
            try:
                response = self.bluemix_connection.create_ranker(training_data=train_file, name=ranker_name)
                self.logger.info("Training request submitted successfully for ranker id:<<%s>>" % response['ranker_id'])
                if self.logger.isEnabledFor(logging.DEBUG):
                    print(json.dumps(response, indent=2))
                return response['ranker_id']
            except WatsonException as ex:
                self.logger.error("Training failed with response: %s" % ex)

                # Check if quota is full & make space if deletion is enabled
                if self._error_due_to_filled_quota(ex):
                    if is_enabled_make_space:
                        self.logger.warn("Quota is full. Deleting all previous rankers to make space.")
                        self.delete_existing_rankers()
                        train_file.seek(0)  # rewind the file handler so that we can make another request
                        return self.train_ranker(train_file_location, False)
                    else:
                        self.logger.error("Quota is full. Use the '-r' parameter to make space by deleting all previous"
                                          " rankers.")
                        raise
                else:
                    raise

    @staticmethod
    def _error_due_to_filled_quota(ex):
        is_quota_error = False
        try:
            if "This user or service instance has the maximum number of rankers" in ex.args[0]:
                is_quota_error = True
        except Exception:
            pass
        return is_quota_error

    def delete_existing_rankers(self):
        """
        Helper method deletes pre-existing rankers under this bluemix url for this user. Expects at least one
        pre-existing ranker to be found.
        """
        previously_created_rankers = self.bluemix_connection.list_rankers()['rankers']

        self.logger.debug("Found %d previously created rankers" % len(previously_created_rankers))
        for ranker in previously_created_rankers:
            response = self.bluemix_connection.delete_ranker(ranker['ranker_id'])
            if self.logger.isEnabledFor(logging.DEBUG):
                print(json.dumps(response, indent=2))

        self.logger.info("Deleted %d rankers successfully" % len(previously_created_rankers))

    def check_ranker_status(self, ranker_id):
        response = self.bluemix_connection.get_ranker_status(ranker_id=ranker_id)
        return response['status'].upper()

    def wait_for_training_to_complete(self, ranker_id):
        """
        Polls (every 30s) the service to check when the ranker's status is no longer "TRAINING".
        Raises exception if the final state is not "AVAILABLE", otherwise returns cleanly.

        :param str ranker_id: the ranker id whose status needs to be checked.
        """
        self.logger.info("Checking/Waiting for training to complete for ranker %s" % ranker_id)

        response = self.bluemix_connection.get_ranker_status(ranker_id=ranker_id)
        num_failed_connections = 0
        while response['status'].upper() == "TRAINING":
            self.logger.debug(
                "Ranker still in status: %s. Will continue polling every %d secs." % (
                    response['status'], self._SECONDS_BETWEEN_STATUS_POLLING_REQUESTS))
            sleep(self._SECONDS_BETWEEN_STATUS_POLLING_REQUESTS)
            try:
                response = self.bluemix_connection.get_ranker_status(ranker_id=ranker_id)
                if self.logger.isEnabledFor(logging.DEBUG):
                    print(json.dumps(response, indent=2))
            except requests.ConnectionError as ex:
                num_failed_connections += 1
                self._handle_exception(ex, num_failed_connections, "Failure %d to get status of ranker <%s>" %
                                       (num_failed_connections, ranker_id))
        self.logger.info("Finished waiting for ranker <<%s>> to train: %s" % (ranker_id, response['status']))

        if response['status'].upper() != "AVAILABLE":
            raise RuntimeError("Unusable ranker, training failed with description: %s" % response['status_description'])
