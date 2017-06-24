import logging
from shutil import rmtree
from tempfile import mkdtemp, NamedTemporaryFile
from time import sleep
from unittest import TestCase
from collections import defaultdict
import requests
from pkg_resources import resource_filename

from examples.upload_documents_to_discovery_collection import document_corpus_as_iterable
from rnr_debug_helpers.utils.discovery_wrappers import DiscoveryProxy, initialize_discovery_service, \
    search_for_byod_environment_id, get_discovery_credentials
from rnr_debug_helpers.utils.io_helpers import initialize_logger, insert_modifier_in_filename, smart_file_open, \
    RankerRelevanceFileQueryStream, PredictionReader
from rnr_debug_helpers.utils.predictions import Prediction

EPSILON = 4
# Test data for a new cluster
MOCK_RELEVANCE_FILE = resource_filename('resources.tests', 'relevance_file_for_xml_sample_data.csv')
MOCK_RELEVANCE_FILE_WITH_DUPS = resource_filename('resources.tests', 'relevance_file_for_xml_sample_data_with_dups.csv')
MOCK_DISCOVERY_CONFIG = resource_filename('resources.tests', 'mock_discovery_config.json')
MOCK_COLLECTION_NAME = 'UnitTestCollection'
MOCK_SOLR_DOCS = resource_filename('resources.tests', 'mock_document_corpus.solr.xml')
EXPECTED_STATS = {'num_queries_which_doesnt_have_any_results': 1.0, 'num_questions': 3.0,
                  'num_results_returned': 4.0}
EXPECTED_PREDICTION_DETAILS_BY_LINE = [Prediction(aid='4', qid='1', rank_score=1.246842, conf_score=None),
                                       Prediction(aid='3', qid='1', rank_score=1.108304, conf_score=None),
                                       Prediction(aid='3', qid='2', rank_score=1.2452893, conf_score=None),
                                       Prediction(aid='4', qid='2', rank_score=0.3962496, conf_score=None)]
NUM_PREDICTIONS = 2
EXPECTED_CORPUS_SIZE = 5


class TestDiscoveryProxy(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment_id = search_for_byod_environment_id(initialize_discovery_service())

    def setUp(self):
        self.given = self
        self.then = self
        self.when = self
        self._and = self

        self.logger = initialize_logger(logging.INFO, name=TestDiscoveryProxy.__name__)

        self.collection_id = None
        self.config_id = None
        self.temp_dir = mkdtemp()

    def tearDown(self):
        self._delete_test_collection_and_configs()
        rmtree(self.temp_dir, ignore_errors=True)

    def _delete_test_collection_and_configs(self):
        discovery = initialize_discovery_service()

        if self.collection_id is not None:
            try:
                discovery.delete_collection(environment_id=self.environment_id,
                                            collection_id=self.collection_id)
                self.logger.info("Successfully deleted test collection: %s" % self.collection_id)
            except Exception as ex:
                self.logger.info('No cleanup required for collection id: %s (got response: %s)' %
                                 (self.collection_id, ex))

        if self.config_id is not None:
            try:

                discovery.delete_configuration(environment_id=self.environment_id,
                                               configuration_id=self.config_id)
                self.logger.info("Successfully deleted test config: %s" % self.config_id)
            except Exception as ex:
                self.logger.info('No cleanup required for config id: %s (got response: %s)' %
                                 (self.config_id, ex))

    def test_config_upload(self):
        self.given.a_discovery_proxy()
        self.when.i_upload_config()
        self.then.i_get_a_valid_config_id()

    def test_create_collection(self):
        """
        TODO: split into multiple tests, running as single test for speed
        """
        self.given.a_discovery_proxy()
        self._and.a_config_id()
        self._and.a_collection_name()
        self.when.i_setup_collection()
        self.then.i_get_a_valid_collection_id()
        self._and.given.a_doc_corpus()
        self.when.i_upload_docs()
        self.then.i_can_see_docs_in_collection()
        self._and.given.a_relevance_file(MOCK_RELEVANCE_FILE)
        self._and.a_prediction_output_path()
        self.when.i_generate_predictions()
        self.then.i_expect_predictions_to_match()
        self._and.given.a_relevance_file(MOCK_RELEVANCE_FILE_WITH_DUPS)
        self.when.i_upload_training_data()
        self.then.i_can_see_training_data_in_collection()

    def i_expect_predictions_to_match(self):
        with smart_file_open(self.prediction_file) as infile:
            reader = PredictionReader(infile)
            i = 0
            expected_rank_scores_by_query = defaultdict(list)
            actual_rank_scores_by_query = defaultdict(list)
            for i, prediction in enumerate(reader):
                self.assertTrue(i < len(EXPECTED_PREDICTION_DETAILS_BY_LINE),
                                msg="expected %d rows of predictions, found %d" %
                                    (len(EXPECTED_PREDICTION_DETAILS_BY_LINE), i + 1))

                # qids and aids should match exactly
                self.assertEquals(EXPECTED_PREDICTION_DETAILS_BY_LINE[i].qid, prediction.qid)
                self.assertEquals(EXPECTED_PREDICTION_DETAILS_BY_LINE[i].aid, prediction.aid)

                # absolute value of rank scores change over time, so save them so we can compare the ordering
                expected_rank_scores_by_query[prediction.qid].append(EXPECTED_PREDICTION_DETAILS_BY_LINE[i].rank_score)
                actual_rank_scores_by_query[prediction.qid].append(prediction.rank_score)

        self.assertTrue(i == (len(EXPECTED_PREDICTION_DETAILS_BY_LINE) - 1),
                        msg="expected %d rows of predictions, found %d" %
                            (len(EXPECTED_PREDICTION_DETAILS_BY_LINE), i + 1))
        for query, expected_rank_scores in expected_rank_scores_by_query.items():
            self.assertEqual([i[0] for i in sorted(enumerate(expected_rank_scores), key=lambda x:x[1])],
                             [i[0] for i in sorted(enumerate(actual_rank_scores_by_query[query]), key=lambda x: x[1])],
                             msg='Expected the ordering of answer ids dictated by the rank orders to match')

    def i_can_see_training_data_in_collection(self):

        actual_examples = self._parse_uploaded_examples_from_collection()
        expected_examples = {}
        with smart_file_open(self.relevance_file) as infile:
            for query in RankerRelevanceFileQueryStream(infile):
                if query.get_qid() in expected_examples:
                    expected_examples[query.get_qid()] += [doc_id for doc_id in query.get_answer_ids() if
                                                           query.get_label(doc_id) > 0]
                    expected_examples[query.get_qid()] = sorted(list(set(expected_examples[query.get_qid()])))
                else:
                    expected_examples[query.get_qid()] = sorted([doc_id for doc_id in query.get_answer_ids() if
                                                                 query.get_label(doc_id) > 0])
        self.assertDictEqual(actual_examples, expected_examples)

    def _parse_uploaded_examples_from_collection(self):
        discovery_creds = get_discovery_credentials()
        response = requests.get(
            "{}/v1/environments/{}/collections/{}/training_data".format(discovery_creds['url'],
                                                                        self.environment_id,
                                                                        self.collection_id),
            params={'version': discovery_creds['version']},
            auth=(discovery_creds['username'], discovery_creds['password']),
            headers={'x-global-transaction-id': 'Rishavs app',
                     'Content-type': 'application/json'})
        response.raise_for_status()
        return {example['natural_language_query']: sorted([doc['document_id'] for doc in example['examples']]) for
                example in response.json()['queries']}

    def i_generate_predictions(self):
        with smart_file_open(self.relevance_file) as infile:
            self.discovery_proxy.generate_natural_language_prediction_scores(
                test_questions=RankerRelevanceFileQueryStream(infile), prediction_file_location=self.prediction_file,
                num_rows=NUM_PREDICTIONS, collection_id=self.collection_id)

    def i_upload_training_data(self):
        with smart_file_open(self.relevance_file) as infile:
            self.discovery_proxy.add_training_data(labelled_query_stream=RankerRelevanceFileQueryStream(infile),
                                                   collection_id=self.collection_id)

    def a_discovery_proxy(self):
        self.discovery_proxy = DiscoveryProxy()

    def i_get_a_valid_config_id(self):
        response = initialize_discovery_service().get_configuration(self.environment_id, self.config_id)
        self.assertEquals(response['configuration_id'], self.config_id)

    def a_config_id(self):
        self.i_upload_config()

    def i_upload_config(self, ):
        self.config_id = self.discovery_proxy.upload_config_file(MOCK_DISCOVERY_CONFIG)

    def a_collection_name(self):
        self.collection_name = MOCK_COLLECTION_NAME

    def i_setup_collection(self):
        self.collection_id = self.discovery_proxy.setup_collection(collection_id=self.collection_name,
                                                                   config_id=self.config_id)

    def a_doc_corpus(self):
        self.doc_corpus = document_corpus_as_iterable(MOCK_SOLR_DOCS)

    def i_get_a_valid_collection_id(self):
        response = initialize_discovery_service().get_collection(environment_id=self.environment_id,
                                                                 collection_id=self.collection_id)
        self.assertEquals(response['configuration_id'], self.config_id)
        self.assertEquals(response['collection_id'], self.collection_id)

        self.assertTrue(self.discovery_proxy.collection_previously_created(self.collection_id))

    def i_upload_docs(self):
        self.response = self.discovery_proxy.upload_documents(collection_id=self.collection_id, corpus=self.doc_corpus,
                                                              max_concurrent_child_processes=3)

    def i_can_see_docs_in_collection(self):
        sleep(2)  # give discovery a chance to process
        num_docs = self.discovery_proxy.get_num_docs_in_collection(self.collection_id)['available'] + \
                   self.discovery_proxy.get_num_docs_in_collection(self.collection_id)['processing']
        self.assertEquals(EXPECTED_CORPUS_SIZE, num_docs)

    def a_relevance_file(self, relevance_file):
        self.relevance_file = relevance_file

    def a_prediction_output_path(self):
        self.prediction_file = insert_modifier_in_filename(NamedTemporaryFile(dir=self.temp_dir).name,
                                                           'prediction_file', 'txt')
