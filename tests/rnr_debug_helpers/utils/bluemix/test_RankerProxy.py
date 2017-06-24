"""
@author: rchakravarti
@creation-date: 1/16/17
"""
import json
import logging
from os import path, symlink
from shutil import rmtree
from tempfile import NamedTemporaryFile, mkdtemp
from unittest import TestCase

import requests
from pkg_resources import resource_filename

from rnr_debug_helpers.compute_ranking_stats import compute_performance_stats
from rnr_debug_helpers.utils.rnr_wrappers import RankerProxy, get_rnr_credentials
from rnr_debug_helpers.utils.io_helpers import initialize_logger, load_config, insert_modifier_in_filename, \
    smart_file_open, \
    RankerRelevanceFileQueryStream, PredictionReader

# Test data for an existing cluster
EPSILON = 4
# Test data for a new cluster
TRAIN_DATA_WITH_AIDS = resource_filename('resources.tests', 'train.aid.csv')
TEST_DATA_WITH_AIDS = resource_filename('resources.tests', 'test.aid.csv')
TEST_DATA_GT = resource_filename('resources.tests', 'test.wea.csv')
TRAIN_DATA_WITH_NO_AID = resource_filename('resources.tests', 'train.no_aid.csv')
EXPECTED_ACCURACY = {
    "ndcg@5": 0.9081918011004686,
    "num_instances": 600.0,
    "num_questions": 30.0,
    'num_queries_predicted': 30,
    "num_top_answers_correct": 29.0,
    'num_queries_skipped_in_preds_file': 0,
    "average_precision_5_truncated": 1.0,
    "recall@5": 0.9699248120300752,
    "top-1-accuracy": 0.9666666666666667
}


class TestRankerProxy(TestCase):
    def setUp(self):
        self.given = self
        self.then = self
        self.when = self
        self._and = self

        self.logger = initialize_logger(logging.INFO, name=TestRankerProxy.__name__)
        self.ranker_id = None
        self.temp_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.temp_dir, ignore_errors=True)
        if self.ranker_id is not None:
            self._try_deleting_ranker(self.ranker_id)

    def _try_deleting_ranker(self, ranker_id):
        config = load_config()
        url, user_id, password = get_rnr_credentials(config)

        response = requests.get(path.join(url, 'v1/rankers', ranker_id),
                                auth=(user_id, password),
                                headers={'x-global-transaction-id': 'Rishavs app',
                                         'Content-type': 'application/json'})
        response_text = json.dumps(response.json(), indent=4, sort_keys=True)
        if response.status_code == 200:
            self.logger.info('Found a test ranker that needs cleanup: %s' % response_text)
            response = requests.delete(path.join(url, 'v1/rankers', ranker_id),
                                       auth=(user_id, password),
                                       headers={'x-global-transaction-id': 'Rishavs app',
                                                'Content-type': 'application/json'})
            response.raise_for_status()
            self.logger.info("Successfully deleted test ranker: %s" % ranker_id)
        else:
            self.logger.info('No cleanup required for ranker id: %s (got response: %s)' % (ranker_id, response_text))

    def test_ranker_training_with_feature_file_that_has_answer_id_column(self):
        self.given.a_ranker_service_connection()
        self._and.a_train_file(TRAIN_DATA_WITH_AIDS)
        self._and.a_ranker_name('unit-test-ranker')
        self._and.answer_id_flag_set_to(True)
        self.when.i_train_a_ranker()
        self.then.i_get_ranker_id()
        self._and.i_can_wait_for_ranker_completion()
        self._and.a_prediction_output_file_location()
        self._and.i_can_generate_predictions_with()
        self._and.predictions_accuracy_matches_expected()

    def test_ranker_training_with_feature_file_with_no_answer_id_column(self):
        self.given.a_ranker_service_connection()
        self._and.a_train_file(TRAIN_DATA_WITH_NO_AID)
        self._and.a_ranker_name('unit-test-ranker')
        self._and.answer_id_flag_set_to(False)
        self.when.i_train_a_ranker()
        self.then.i_get_ranker_id()
        self._and.i_can_wait_for_ranker_completion()

    def a_prediction_output_file_location(self):
        self.prediction_file = insert_modifier_in_filename(NamedTemporaryFile(delete=False, dir=self.temp_dir).name,
                                                           'prediction_file', 'txt')

    def a_train_file(self, train_file):
        # we actually use a symlink because we might generate more files in the same directory
        self.train_file = insert_modifier_in_filename(NamedTemporaryFile(delete=False, dir=self.temp_dir).name,
                                                      'train_file', 'csv')
        symlink(train_file, self.train_file)

    def i_can_generate_predictions_with(self):
        self.ranker_service.generate_ranker_predictions(ranker_id=self.ranker_id,
                                                        test_file_location=TEST_DATA_WITH_AIDS,
                                                        prediction_file_location=self.prediction_file)

    def a_ranker_service_connection(self):
        self.ranker_service = RankerProxy()

    def i_train_a_ranker(self):
        self.ranker_id = self.ranker_service.train_ranker(train_file_location=self.train_file,
                                                          ranker_name=self.ranker_name,
                                                          is_enabled_make_space=True,
                                                          train_file_has_answer_id=self.has_answer_id_flag)

    def a_ranker_name(self, ranker_name):
        self.ranker_name = ranker_name

    def answer_id_flag_set_to(self, flag):
        self.has_answer_id_flag = flag

    def i_get_ranker_id(self):
        self.assertIsNotNone(self.ranker_id)
        self.assertTrue(isinstance(self.ranker_id, str))

    def i_can_wait_for_ranker_completion(self):
        self.ranker_service.wait_for_training_to_complete(self.ranker_id)

    def predictions_accuracy_matches_expected(self):
        self.assertTrue(path.isfile(self.prediction_file))
        with smart_file_open(TEST_DATA_GT) as infile:
            ground_truth_reader = RankerRelevanceFileQueryStream(infile)
            with smart_file_open(self.prediction_file) as preds_file:
                prediction_reader = PredictionReader(preds_file, file_has_answer_ids=True,
                                                     file_has_confidence_scores=True)
                self.actual_stats, _ = compute_performance_stats(prediction_reader=prediction_reader,
                                                                 ground_truth_query_stream=ground_truth_reader,
                                                                 k=5)

        for stat_type in ['average_precision_5_truncated', 'top-1-accuracy', 'num_questions', 'ndcg@5', 'num_instances',
                          'num_queries_predicted', 'num_queries_skipped_in_preds_file']:
            self.assertAlmostEqual(EXPECTED_ACCURACY[stat_type], self.actual_stats[stat_type], delta=EPSILON)
