"""
@author: rchakravarti
@creation-date: 1/16/17
"""
import json
import logging
from os import path
from shutil import rmtree
from tempfile import mkdtemp, NamedTemporaryFile
from unittest import TestCase

import requests
from pkg_resources import resource_filename

from rnr_debug_helpers.utils.rnr_wrappers import RetrieveAndRankProxy, get_rnr_credentials
from rnr_debug_helpers.utils.io_helpers import initialize_logger, load_config, insert_modifier_in_filename, \
    smart_file_open, RankerRelevanceFileQueryStream, PredictionReader
from rnr_debug_helpers.utils.predictions import Prediction

# Test data for an existing cluster

MOCK_QUERY_WITH_RESULTS = 'experimental study'
MOCK_QUERY_WITH_ZERO_RESULTS = 'supercalifragilisticexpialidocious'
EXPECTED_HEADER = 'answer_id,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,r1,r2,s'
EXPECTED_FEATURES = [
    [1, 0.12695485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010117257, 0.0, 0.0, 0.0, 0.07283872, 0.0,
     0.0, 0.0, 1.0, 0, 0.6931471805599453, 0.20991084],
    [2, 0.021270415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
     1, 0.4054651081081644, 0.021270415]]
EPSILON = 4
# Test data for a new cluster
MOCK_SOLR_CONFIG = resource_filename('resources.tests', 'sample_solr_config.zip')
MOCK_SOLR_DOCS = resource_filename('resources.tests', 'sample_data.json')
NUM_DOCS_IN_SAMPLE_DATA = 5
RELEVANCE_FILE_FOR_SAMPLE_DATA = resource_filename('resources.tests', 'relevance_file_for_sample_data.csv')
EXPECTED_STATS = {'num_queries_which_doesnt_have_any_results': 1.0, 'num_questions': 3.0,
                  'num_results_returned': 4.0}
EXPECTED_PREDICTION_DETAILS_BY_LINE = [Prediction(aid='1', qid='1', rank_score=0.20991084, conf_score=None),
                                       Prediction(aid='2', qid='1', rank_score=0.021270415, conf_score=None),
                                       Prediction(aid='2', qid='2', rank_score=0.021270415, conf_score=None),
                                       Prediction(aid='1', qid='2', rank_score=0.0, conf_score=None)]


class TestRetrieveAndRankProxy(TestCase):
    def setUp(self):
        self.given = self
        self.then = self
        self.when = self
        self._and = self

        self.logger = initialize_logger(logging.INFO, name=TestRetrieveAndRankProxy.__name__)
        self.new_cluster_id = None
        self.temp_dir = mkdtemp()

    def tearDown(self):
        if self.new_cluster_id is not None:
            self._delete_test_cluster(self.new_cluster_id)
        rmtree(self.temp_dir, ignore_errors=True)

    def _delete_test_cluster(self, cluster_id):
        self.logger.info("Attempting to clean up the test cluster that was spun up for the unit test: %s" % cluster_id)
        config = load_config()
        url, user_id, password = get_rnr_credentials(config)

        response = requests.get(path.join(url, 'v1/solr_clusters', cluster_id),
                                auth=(user_id, password),
                                headers={'x-global-transaction-id': 'Rishavs app',
                                         'Content-type': 'application/json'})
        response_text = json.dumps(response.json(), indent=4, sort_keys=True)
        if response.status_code == 200:
            self.logger.info('Found a test cluster that needs cleanup: %s' % response_text)
            response = requests.delete('%s/v1/solr_clusters/%s' % (url, cluster_id),
                                       auth=(user_id, password),
                                       headers={'x-global-transaction-id': 'Rishavs app',
                                                'Content-type': 'application/json'})
            response.raise_for_status()
            self.logger.info("Successfully deleted test cluster: %s" % cluster_id)
        else:
            self.logger.info('No cleanup required for cluster id: %s (got response: %s)' % (cluster_id, response_text))

    def i_can_run_tests_to_validate_feature_generation(self):
        # test mock query with some results and header turned off
        self.when.i_call_get_fcselect_features(query=MOCK_QUERY_WITH_RESULTS, num_results=2, generate_header=False)
        self.then.i_expect_num_results_to_match(expected_num_results=2, generate_header=False)
        self._and.i_expect_features_vectors_for_mock_query(expected_num_results=2, generate_header=False)

        # test mock query with some results and header turned on
        self.when.i_call_get_fcselect_features(query=MOCK_QUERY_WITH_RESULTS, num_results=1, generate_header=True)
        self.then.i_expect_num_results_to_match(expected_num_results=1, generate_header=True)
        self._and.i_expect_features_vectors_for_mock_query(expected_num_results=1, generate_header=True)

        # test query with no results
        self.when.i_call_get_fcselect_features(query=MOCK_QUERY_WITH_ZERO_RESULTS, num_results=10, generate_header=True)
        self.then.i_expect_num_results_to_match(expected_num_results=0, generate_header=False)

        # test prediction file generation
        self.given.a_prediction_output_path()
        self._and.a_test_relevance_file()
        self.when.i_generate_predictions()
        self.then.prediction_file_should_exist()
        self._and.predictions_should_match_expected()
        self._and.i_expect_stats_to_match_expected()

    def test_setup_new_cluster_and_collection(self):
        """
        TODO: currently runs all tests using a single cluster for speed...probably should split this massive
        unit test up
        """
        self.given.a_bluemix_cluster()
        self._and.a_solr_config()
        self._and.a_solr_config_id('TestConfig')
        self._and.a_collection_id('TestCreateCollection')
        self.when.i_setup_cluster_and_collection()
        self.then.collection_exists()
        self._and.a_config_exists()
        self._and.collection_num_docs_is(0)
        self._and.i_can_upload_docs()
        self._and.see_docs_are_in_collection()
        self._and.i_can_run_tests_to_validate_feature_generation()
        self._and.if_i_reinitialize_cluster_with_generated_id()
        self.then.see_docs_are_in_collection()

    def prediction_file_should_exist(self):
        self.assertTrue(path.isfile(self.prediction_file))

    def predictions_should_match_expected(self):
        with smart_file_open(self.prediction_file) as infile:
            reader = PredictionReader(infile)
            i = 0
            for i, prediction in enumerate(reader):
                self.assertTrue(i < len(EXPECTED_PREDICTION_DETAILS_BY_LINE),
                                msg="expected %d rows of predictions, found %d" %
                                    (len(EXPECTED_PREDICTION_DETAILS_BY_LINE), i + 1))
                self.assertEquals(EXPECTED_PREDICTION_DETAILS_BY_LINE[i], prediction)

        self.assertTrue(i == (len(EXPECTED_PREDICTION_DETAILS_BY_LINE) - 1),
                        msg="expected %d rows of predictions, found %d" %
                            (len(EXPECTED_PREDICTION_DETAILS_BY_LINE), i + 1))

    def i_expect_stats_to_match_expected(self):
        self.assertIsNotNone(self.stats)
        for stat_type in EXPECTED_STATS.keys():
            self.assertAlmostEquals(EXPECTED_STATS[stat_type], self.stats[stat_type])

    def i_generate_predictions(self):
        with smart_file_open(self.test_relevance_file) as infile:
            self.stats = self.bluemix_cluster.generate_fcselect_prediction_scores(
                test_questions=RankerRelevanceFileQueryStream(infile),
                prediction_file_location=self.prediction_file, collection_id=self.collection_id)

    def a_prediction_output_path(self):
        self.prediction_file = insert_modifier_in_filename(NamedTemporaryFile(dir=self.temp_dir).name, 'prediction',
                                                           'txt')

    def a_test_relevance_file(self):
        self.test_relevance_file = RELEVANCE_FILE_FOR_SAMPLE_DATA

    def if_i_reinitialize_cluster_with_generated_id(self):
        self.a_bluemix_cluster(solr_cluster_id=self.new_cluster_id)

    def a_bluemix_cluster(self, solr_cluster_id=None):
        if solr_cluster_id is None:
            self.logger.info('Testing with a new solr cluster')
            self.bluemix_cluster = RetrieveAndRankProxy(cluster_name='UnitTest', cluster_size=1)
            self.new_cluster_id = self.bluemix_cluster.solr_cluster_id
        else:
            self.logger.info('Testing with existing test cluster: %s' % solr_cluster_id)
            self.bluemix_cluster = RetrieveAndRankProxy(solr_cluster_id=solr_cluster_id)

    def a_solr_config(self):
        self.solr_config = MOCK_SOLR_CONFIG

    def a_collection_id(self, collection_id):
        self.collection_id = collection_id

    def a_solr_config_id(self, config_id):
        self.solr_config_id = config_id

    def i_setup_cluster_and_collection(self):
        self.bluemix_cluster.setup_cluster_and_collection(config_id=self.solr_config_id, config_zip=self.solr_config,
                                                          collection_id=self.collection_id)

    def collection_exists(self):
        self.assertTrue(self.bluemix_cluster.collection_previously_created(self.collection_id))

    def collection_num_docs_is(self, expected_num_docs):
        self.assertEquals(expected_num_docs, self.bluemix_cluster.get_num_docs_in_collection(self.collection_id))

    def a_config_exists(self):
        self.bluemix_cluster.config_previously_uploaded(self.solr_config_id)

    def i_can_upload_docs(self):
        self.bluemix_cluster.upload_documents_to_collection(collection_id=self.collection_id,
                                                            corpus_file=MOCK_SOLR_DOCS, content_type='application/json')

    def see_docs_are_in_collection(self):
        self.collection_num_docs_is(NUM_DOCS_IN_SAMPLE_DATA)

    def i_call_get_fcselect_features(self, query, num_results, generate_header):
        self.response = self.bluemix_cluster.get_fcselect_features(query, num_results_to_return=num_results,
                                                                   generate_header=generate_header,
                                                                   collection_id=self.collection_id)

    def i_expect_num_results_to_match(self, expected_num_results, generate_header):
        if generate_header:
            self.assertEquals(expected_num_results + 1, len(self.response),
                              "Expected %d results plus header row, but found %d: %s" % (
                                  expected_num_results, len(self.response), self.response))
        else:
            self.assertEquals(expected_num_results, len(self.response),
                              "Expected %d results, but found %d: %s" % (
                                  expected_num_results, len(self.response), self.response))

    def i_expect_features_vectors_for_mock_query(self, expected_num_results, generate_header):
        if generate_header:
            self.assertEquals(EXPECTED_HEADER, ",".join(self.response.pop(0)))

        for i in range(expected_num_results):
            try:
                actual_vector = [float(x) for x in self.response[i]]
            except Exception as ex:
                self.logger.error('Unable to parse row %d of the response %s for feature vectors: %s' %
                                  (i, self.response, ex))
                raise

            self.assertEquals(EXPECTED_FEATURES[i], actual_vector)
