"""
Just holds some constants
@author: rchakravarti
@creation-date: 9/20/16
"""
import csv
import json
import logging
import sys
import time
import urllib.parse
from collections import defaultdict
from datetime import timedelta
from multiprocessing import Manager, Process, active_children
from os import path
from pprint import pprint
from shutil import move
from warnings import warn

import requests
from requests.adapters import HTTPAdapter
from watson_developer_cloud import discovery_v1

from rnr_debug_helpers.utils.io_helpers import load_config, initialize_logger, smart_file_open, get_temp_file, \
    initialize_retry_settings
from rnr_debug_helpers.utils.predictions import Prediction

# only applicable if you don't use the natural_language_query parameter
CHARS_THAT_HAVE_TO_BE_ESCAPED_FOR_DISCOVERY = ['\\', ':', '!', '"', '~', '(', ')', '[', ']', '|', ',', '>', '=',
                                               '<', '^', '*']
DOC_ID_FIELD_NAME = 'id'


def escape_special_tokens_for_discovery_syntax(text):
    """
    # only applicable if you don't use the natural_language_query parameter
    :param text:
    :return:
    """
    modified_text = text
    for char in CHARS_THAT_HAVE_TO_BE_ESCAPED_FOR_DISCOVERY:
        modified_text = modified_text.replace(char, "\%s" % char)

    return modified_text


def get_discovery_credentials(config=load_config()):
    url = config.get('Discovery', 'url', fallback=discovery_v1.default_url)
    user = config.get('Discovery', 'user')
    password = config.get('Discovery', 'password')
    version = config.get('Discovery', 'version', fallback=discovery_v1.latest_version)
    return {'url': url, 'version': version, 'username': user, 'password': password}


def initialize_discovery_service(config=load_config()):
    return discovery_v1.DiscoveryV1(**get_discovery_credentials(config))


def search_for_byod_environment_id(discovery):
    """
    From the documentation, it seems like there can only be one `environment` for each set of Discovery credentials
    where you can bring your own data (byod) and create collections.  So we can try to search for such an
    environment if an environment_id wasn't explicitly provided.
    :param discovery_v1.Discovery discovery:
    :return: the first environment_id with the `name` == "byod"
    :rtype: str
    """
    for environment in discovery.get_environments()['environments']:
        if environment['name'] == 'byod':
            return environment['environment_id']
    raise RuntimeError("Need to specify an environment id, couldn't find one in the config...")


class DiscoveryProxy(object):
    """
    My convenience wrapper for the Discovery API calls as well as helper methods to pre-process/post-process
        data going into and out of those API calls.
    """

    def __init__(self, config=load_config(), logger=initialize_logger(logging.INFO, 'DiscoveryProxy')):
        """
        Initialize the connection to Bluemix

        :param CofigParser config: An initialized config with user, password and environment id
        """
        self.config = config
        self.logger = logger
        self.discovery = initialize_discovery_service(config)
        self.environment_id = config.get('Discovery', 'environment_id',
                                         fallback=search_for_byod_environment_id(self.discovery))
        self.http_connection = requests.Session()
        self.http_connection.auth = (config.get('Discovery', 'user'), config.get('Discovery', 'password'))
        self.http_connection.headers = {'x-global-transaction-id': 'rnr-tuning-scripts'}
        self.http_connection.mount('https://', HTTPAdapter(max_retries=initialize_retry_settings(config)))
        self.http_connection.mount('http://', HTTPAdapter(max_retries=initialize_retry_settings(config)))

    @property
    def environment_id(self):
        return self._environment_id

    @environment_id.setter
    def environment_id(self, val):
        # test the environment id before setting it
        self.discovery.get_environment(environment_id=val)
        self._environment_id = val

    def _get_runtime_predictions(self, question_number, query_text, collection_id, num_results_to_return):
        query_parameters = {'natural_language_query': query_text.encode('utf-8'),
                            'count': num_results_to_return,
                            'return': 'score,%s' % DOC_ID_FIELD_NAME,
                            'version': self.discovery.version}

        response = None
        try:
            response = self.http_connection.get("{}/v1/environments/{}/collections/{}/query".format(self.discovery.url,
                                                                                                    self.environment_id,
                                                                                                    collection_id),
                                                params=urllib.parse.urlencode(query_parameters))
            response.raise_for_status()
            predictions = self._parse_response_content_for_predictions(question_number, response)
        except Exception as ex:
            self.logger.error("Error completing natural_language_query <<%s>>: %s" % (query_text, str(ex)))
            if response is not None:
                pprint(vars(response))
            raise
        return predictions, response.elapsed

    def generate_natural_language_prediction_scores(self, test_questions, prediction_file_location, collection_id,
                                                    num_rows=10):
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
        """
        self.logger.info(
            "Sending runtime requests from <<%s>> to collection: <<%s>> (predictions will be written to: <<%s>>)" % (
                test_questions, collection_id, prediction_file_location))

        temp_file = get_temp_file(prediction_file_location)
        stats = defaultdict(float)
        with smart_file_open(temp_file, 'w') as prediction_outfile:
            writer = csv.writer(prediction_outfile, delimiter=' ')
            for query in test_questions:
                stats['num_questions'] += 1
                self.logger.debug("Generate predictions for query <<%s>>" % query.get_qid())
                predictions, response_time = self._get_runtime_predictions(stats['num_questions'],
                                                                           query_text=query.get_qid(),
                                                                           collection_id=collection_id,
                                                                           num_results_to_return=num_rows)
                stats['response_time_in_seconds'] += response_time.total_seconds()
                if predictions:
                    stats['num_results_returned'] += len(predictions)
                    self._write_results_to_file(predictions, writer)
                else:
                    stats['num_queries_which_doesnt_have_any_results'] += 1
                if self.logger.isEnabledFor(logging.DEBUG) or stats['num_questions'] % 500 == 0:
                    self.logger.info('Generated predictions for %d queries' % stats['num_questions'])

            if stats['num_questions'] < 1:
                raise ValueError("No test instances found in the file")
            stats['avg_response_time_in_seconds'] = stats['response_time_in_seconds'] / stats['num_questions']

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
        if response_content["matching_results"] > 0:
            for doc in response_content['results']:
                results.append(Prediction(qid=question_number, aid=doc[DOC_ID_FIELD_NAME], rank_score=doc['score'],
                                          conf_score=None))
        else:
            self.logger.warn('Empty response: %s' % vars(response))
        return results

    def setup_collection(self, collection_id="NewCollection",
                         collection_description='TestCollection', config_id=None, config_file=None):
        """
        If the collection id is not already present in the cluster, creates new ones.
        Requires either the config id or the config file in that case
        :param str or None collection_id:
        :param str or None collection_description:
        :param str or None config_id:
        :param str or None config_file:
        """
        if not self.collection_previously_created(collection_id):
            self.logger.info('Collection id not found, creating a new one')
            if config_id is None:
                self.logger.info('Config id not found, uploading a new one')
                config_id = self.upload_config_file(config_file)
                self.logger.info('Uploaded a new config: %s' % config_id)

            response = self.discovery.create_collection(environment_id=self.environment_id, configuration_id=config_id,
                                                        description=collection_description, name=collection_id)
            collection_id = response["collection_id"]
            self.logger.info('Created collection: %s' % collection_id)
        return collection_id

    def print_collection_stats(self, collection_id):
        json.dump(self.discovery.get_collection(environment_id=self.environment_id, collection_id=collection_id),
                  sys.stdout, sort_keys=True, indent=4)

    def collection_previously_created(self, collection_id):
        response = self.discovery.list_collections(self.environment_id)
        return collection_id in [c["collection_id"] for c in response['collections']]

    def upload_config_file(self, config_filepath):
        if not path.isfile(config_filepath):
            raise ValueError("Missing config file: %s" % config_filepath)

        with smart_file_open(config_filepath) as config_file:
            self.logger.info("Uploading configuration")
            config_status = self.discovery.create_configuration(self.environment_id, config_data=json.load(config_file))
            if self.logger.isEnabledFor(logging.DEBUG):
                json.dump(config_status, sys.stdout, sort_keys=True, indent=4)
        return config_status['configuration_id']

    def get_num_docs_in_collection(self, collection_id):
        return self.discovery.get_collection(environment_id=self.environment_id,
                                             collection_id=collection_id)["document_counts"]

    def _wait_for_processors_to_free_up(self, max_concurrent_processes):
        while len(active_children()) >= max_concurrent_processes:
            self.logger.debug("Waiting a few seconds for processors to free up")
            time.sleep(0.1)

    def upload_documents(self, collection_id, corpus, max_concurrent_child_processes=20):
        """

        :param str collection_id: collection to upload to
        :param Iterable corpus: an iterable which yields (doc_id, doc_as_json)
        :param int max_concurrent_child_processes: the maximum number of concurrent processes that are spawned
          to help parrallelize the document upload requests
        """
        stats = defaultdict(int)

        # Setup manager so we can do multiprocessing to speed things up
        file_processors = list()
        manager = Manager()
        response_from_processors = manager.dict()

        for doc_id, body in corpus:
            stats['num_docs'] += 1
            self._wait_for_processors_to_free_up(max_concurrent_child_processes)

            file_processors.append(Process(target=upload_file_to_discovery_collection,
                                           args=(self.config, self.environment_id, collection_id, doc_id, body,
                                                 response_from_processors)))
            file_processors[-1].start()
            if self.logger.isEnabledFor(logging.DEBUG) or stats['num_docs'] % 1000 == 0:
                self.logger.info('Submitted %d upload requests' % stats['num_docs'])
            stats['num_requests_submitted'] += 1

        self.logger.info('Done submitted requests, checking up on the status of the requests')

        # check for failures
        stats['counts_by_status'] = self._check_file_processes(file_processors, response_from_processors)

        self.logger.info('Processed %d docs' % stats['num_docs'])
        json.dump(stats, sys.stdout, sort_keys=True, indent=4)

    @staticmethod
    def _count_non_zero_exit_codes(child_processes_in_play):
        num_failed_child_processes = 0
        for p in child_processes_in_play:
            p.join()
            if p.exitcode != 0:
                num_failed_child_processes += 1
        return num_failed_child_processes

    def _check_file_processes(self, file_processors, response_from_processors):
        self.logger.info('Of the %d spawned doc upload processes, %d had non-zero exit codes' %
                         (len(file_processors), self._count_non_zero_exit_codes(file_processors)))

        counts_by_status = defaultdict(int)
        failed_files = list()
        for doc_id in response_from_processors.keys():
            response = response_from_processors[doc_id]
            if isinstance(response, Exception):
                counts_by_status['failed'] += 1
                failed_files.append(doc_id)
            else:
                counts_by_status[response] += 1

        if counts_by_status['failed'] > 0:
            self.logger.error("Encountered errors with the following %d files: %s" % (len(failed_files), failed_files))

        return counts_by_status

    def add_training_data(self, labelled_query_stream, collection_id):
        self.logger.info('Adding training data from %s to collection %s' % (labelled_query_stream, collection_id))
        num_queries = 0
        for query in labelled_query_stream:
            training_example_for_discovery = self._format_training_example_from_labelled_query(query)
            try:
                self._upload_training_example(training_example_for_discovery, collection_id)
            except Exception:
                self.logger.error('Error uploading query: %s' % query)
                raise
            finally:
                self.logger.debug('Uploaded %d queries worth of training data' % num_queries)
                num_queries += 1
        self.logger.info('Uploaded %d queries worth of training data' % num_queries)

    def _upload_training_example(self, training_example_for_discovery, collection_id):
        response = self.http_connection.post(
            "{}/v1/environments/{}/collections/{}/training_data".format(self.discovery.url,
                                                                        self.environment_id,
                                                                        collection_id),
            data=json.dumps(training_example_for_discovery),
            params={'version': self.discovery.version},
            headers={'Content-type': 'application/json'})
        if response.status_code == 409:
            self.logger.warn('Encountered this query text a second time, so just adding the examples: %s' %
                             training_example_for_discovery['natural_language_query'])
            query_id, previously_labelled_doc_ids = self. \
                _find_query_id_for_previously_uploaded_example(collection_id, training_example_for_discovery)

            for example in training_example_for_discovery['examples']:
                if example['document_id'] not in previously_labelled_doc_ids:
                    response = self.http_connection.post(
                        "{}/v1/environments/{}/collections/{}/training_data/{}/examples".format(self.discovery.url,
                                                                                                self.environment_id,
                                                                                                collection_id,
                                                                                                query_id),
                        data=json.dumps(example),
                        params={'version': self.discovery.version},
                        headers={'Content-type': 'application/json'})
                    if response.status_code != 200 or self.logger.isEnabledFor(logging.DEBUG):
                        pprint(vars(response))
                    response.raise_for_status()

        elif response.status_code != 201:
            pprint(vars(response))
            response.raise_for_status()

    @staticmethod
    def _format_training_example_from_labelled_query(query):
        training_example = {'natural_language_query': query.get_qid(), 'examples': []}
        for doc_id in query.get_answer_ids():
            if query.get_label(doc_id) > 0:
                # HACK: refer to hack below; since we're not sure how to map the auto-generated document_id to
                # the original one, we try to use the cross_reference query with the fake field we added to the doc
                # with our original doc id
                training_example['examples'].append(
                    {'document_id': doc_id, 'relevance': query.get_label(doc_id),
                     'cross_reference': "%s:%s" % (DOC_ID_FIELD_NAME, doc_id)})
        return training_example

    def _find_query_id_for_previously_uploaded_example(self, collection_id, training_example_for_discovery):
        self.logger.warn('Found more than one labelled query with the same text, consolidating into the same'
                         ' training example for Discovery')
        response = self.http_connection.get(
            "{}/v1/environments/{}/collections/{}/training_data".format(self.discovery.url,
                                                                        self.environment_id,
                                                                        collection_id),
            params={'version': self.discovery.version},
            headers={'Content-type': 'application/json'})
        if response.status_code != 200 or self.logger.isEnabledFor(logging.DEBUG):
            pprint(vars(response))
        response.raise_for_status()

        for query in response.json()['queries']:
            if query['natural_language_query'].lower().strip() == training_example_for_discovery[
                'natural_language_query'].lower().strip():
                return query['query_id'], [example['document_id'] for example in query['examples']]

        raise RuntimeError('Did not find a match in training data in collection %s for query %s' %
                           (collection_id, training_example_for_discovery['natural_language_query']))


def upload_file_to_discovery_collection(config, environment_id, collection_id, doc_id, json_doc, pipe_to_main_process):
    try:
        conn = initialize_discovery_service(config)
        response = conn.query(environment_id=environment_id, collection_id=collection_id,
                              query_options={'query': '%s:%s' % (DOC_ID_FIELD_NAME, doc_id)})
        if response['matching_results'] == 1:
            pipe_to_main_process[doc_id] = {'status': 'PreviouslyUploaded'}
        elif response['matching_results'] == 0:
            # discovery_creds = get_discovery_credentials(config)
            # response = requests.post("{}/v1/environments/{}/collections/{}/documents/{}".format(discovery_creds['url'],
            #                                                                                     environment_id,
            #                                                                                     collection_id, doc_id),
            #                          data=json.dumps(json_doc),
            #                          params={'version': discovery_creds['version']},
            #                          auth=(discovery_creds['username'], discovery_creds['password']),
            #                          headers={'x-global-transaction-id': 'Rishavs app',
            #                                   'Content-type': 'application/json'})
            # response.raise_for_status()
            # response_as_json = response.json()

            # HACK: Currently the document id passed in IS NOT the document id with which Discovery stores the
            # document; instead Discovery seems to over ride the document id with its own auto-generated document
            # id...so we store the old document_id as a field titled DOC_ID_FIELD_NAME in the body of the document
            if DOC_ID_FIELD_NAME in json_doc:
                raise RuntimeError('Currently there is no way to override the auto-generated Discovery document id'
                                   ' with the one we want, so we store the desired document id as field %s in the '
                                   'doc. But seems like the body already contains a field titled %s.  Change the '
                                   'DOC_ID_FIELD_NAME constant in discovery_wrappers.py')
            json_doc[DOC_ID_FIELD_NAME] = doc_id
            response_as_json = conn.add_document(environment_id=environment_id, collection_id=collection_id,
                                                 file_data=json.dumps(json_doc),
                                                 mime_type='application/json')
            if response_as_json[u'status'] != u'processing':
                pprint(vars(response_as_json))

            pipe_to_main_process[doc_id] = response_as_json[u'status']
        else:
            raise ValueError('Unexpected results in collection for id <%s>: %s ' % (doc_id, response))
    except Exception as ex:
        warn('Error uploading doc: %s' % ex)
        pipe_to_main_process[doc_id] = ex
        raise
