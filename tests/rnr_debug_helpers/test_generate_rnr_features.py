"""
@author: rchakravarti
@creation-date: 1/16/17
"""
import csv
import json
import logging
from shutil import rmtree
from tempfile import NamedTemporaryFile, mkdtemp
from unittest import TestCase

import requests
from pkg_resources import resource_filename

from rnr_debug_helpers.generate_rnr_feature_file import generate_rnr_features
from rnr_debug_helpers.utils.rnr_wrappers import RetrieveAndRankProxy, get_rnr_credentials
from rnr_debug_helpers.utils.io_helpers import initialize_logger, load_config, RankerRelevanceFileQueryStream

EPSILON = 4
_AID_COL = 1
_QID_COL = 0
_GT_COL = -1

# Test data for a new cluster
MOCK_SOLR_CONFIG = resource_filename('resources.tests', 'sample_solr_config.zip')
MOCK_SOLR_DOCS = resource_filename('resources.tests', 'sample_data.json')
MOCK_RELEVANCE_FILE = resource_filename('resources.tests', 'relevance_file_for_sample_data.csv')
EXPECTED_HEADER = 'question_id,answer_id,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,' \
                  'r1,r2,s,relevance_label'.split(',')
EXPECTED_FEATURE_VEC_DETAILS_BY_LINE = [{'qid': '1', 'answer_id': '1', 'gt': 2},
                                        {'qid': '1', 'answer_id': '2', 'gt': 1},
                                        {'qid': '2', 'answer_id': '2', 'gt': 1},
                                        {'qid': '2', 'answer_id': '1', 'gt': 0}]
EXPECTED_STATS = {'avg_num_correct_answers_per_query_in_gt_file': 5.0 / 3,
                  'avg_num_correct_answers_per_query_in_rnr_results_default': 3.0 / 2,
                  'avg_num_search_results_retrieved_per_query': 2.0,
                  'num_correct_in_gt_file': 5,
                  'num_correct_in_search_result': 3,
                  'num_occurrences_of_label_1': 2,
                  'num_occurrences_of_label_2': 2,
                  'num_occurrences_of_label_5': 1,
                  'num_queries': 3,
                  'num_queries_where_at_least_correct_answer_didnt_appear_in_rnr': 1,
                  'num_queries_with_atleast_one_search_result': 2,
                  'num_queries_with_zero_rnr_results': 1,
                  'num_search_results_retrieved': 4}


class TestGenerateRnrFeatures(TestCase):
    def setUp(self):
        self.given = self
        self.then = self
        self.when = self
        self._and = self
        self._with = self

        self.logger = initialize_logger(logging.INFO, name=TestGenerateRnrFeatures.__name__)
        self.temp_dir = mkdtemp()
        self.bluemix_cluster = None

    def tearDown(self):
        if self.bluemix_cluster is not None and self.bluemix_cluster.solr_cluster_id is not None:
            self._delete_test_cluster(self.bluemix_cluster.solr_cluster_id)
        rmtree(self.temp_dir, ignore_errors=True)

    def _delete_test_cluster(self, cluster_id):
        self.logger.info("Attempting to clean up the test cluster that was spun up for the unit test: %s" % cluster_id)
        config = load_config()
        url, user_id, password = get_rnr_credentials(config)

        response = requests.get('%s/v1/solr_clusters/%s' % (url, cluster_id),
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

    def test_setup_new_cluster_and_collection(self):
        self.given.a_bluemix_cluster_with_docs()
        self._and.a_set_of_labelled_queries()
        self._and.an_output_file()
        self.when.i_generate_feature_file()
        self.then.i_first_row_is_feature_header()
        self._and.file_contains_expected_features()
        self._and.stats_match_expected()

    def i_first_row_is_feature_header(self):
        self.output_file = open(self.output_file.name, 'r')
        reader = csv.reader(self.output_file)
        self.assertEquals(next(reader), EXPECTED_HEADER)

    def file_contains_expected_features(self):

        feature_rows = [row for row in csv.reader(self.output_file)]
        self.assertEquals(len(feature_rows), len(EXPECTED_FEATURE_VEC_DETAILS_BY_LINE))

        for i, row in enumerate(feature_rows):
            self.assertEquals(EXPECTED_FEATURE_VEC_DETAILS_BY_LINE[i]['qid'], row[_QID_COL])
            self.assertEquals(EXPECTED_FEATURE_VEC_DETAILS_BY_LINE[i]['answer_id'], row[_AID_COL])
            self.assertEquals(EXPECTED_FEATURE_VEC_DETAILS_BY_LINE[i]['gt'], int(row[_GT_COL]))
            for j in range(2, len(EXPECTED_HEADER) - 1):
                self.assertTrue(isinstance(float(row[j]), float))

    def stats_match_expected(self):
        mismatching_stats_message = 'Expected stats: %s\nActual stats: %s' % \
                                    (json.dumps(EXPECTED_STATS, indent=4, sort_keys=True),
                                     json.dumps(self.stats, indent=4, sort_keys=True))

        self.assertEqual(len(self.stats), len(EXPECTED_STATS), msg='Length of stats dont match with expected. %s'
                                                                   % mismatching_stats_message)
        for stat_type in EXPECTED_STATS.keys():
            self.assertAlmostEqual(self.stats[stat_type], EXPECTED_STATS[stat_type], places=EPSILON,
                                   msg='Mismatch in <%s>. %s' % (stat_type, mismatching_stats_message))

    def i_generate_feature_file(self):
        self.stats = generate_rnr_features(cluster_id=self.bluemix_cluster.solr_cluster_id,
                                           collection_id=self.collection_id,
                                           in_query_stream=self.labelled_query_stream, outfile=self.output_file)

    def a_bluemix_cluster_with_docs(self, solr_cluster_id=None):
        if solr_cluster_id is None:
            self.logger.info('Testing with a new solr cluster')
            self.bluemix_cluster = RetrieveAndRankProxy(cluster_name='UnitTest', cluster_size=1)

        else:
            self.logger.info('Testing with existing cluster: %s' % solr_cluster_id)
            self.bluemix_cluster = RetrieveAndRankProxy(solr_cluster_id=solr_cluster_id)

        self.collection_id = 'TestCreateCollection'
        self.bluemix_cluster.setup_cluster_and_collection(config_id='TestConfig', config_zip=MOCK_SOLR_CONFIG,
                                                          collection_id=self.collection_id)
        self.bluemix_cluster.upload_documents_to_collection(collection_id=self.collection_id,
                                                            corpus_file=MOCK_SOLR_DOCS, content_type='application/json')

    def a_set_of_labelled_queries(self):
        self.labelled_query_stream = RankerRelevanceFileQueryStream(fh=open(MOCK_RELEVANCE_FILE, 'r'))

    def an_output_file(self):
        self.output_file = NamedTemporaryFile('w', dir=self.temp_dir, delete=False)
