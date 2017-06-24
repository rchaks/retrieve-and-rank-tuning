"""
@author: rchakravarti
@creation-date: 1/16/17
"""
from unittest import TestCase

from pkg_resources import resource_filename

from rnr_debug_helpers.compute_ranking_stats import compute_performance_stats
from rnr_debug_helpers.utils.io_helpers import smart_file_open, \
    RankerRelevanceFileQueryStream, PredictionReader

# Test data for an existing cluster
EPSILON = 6
# Test data for a new cluster
MOCK_PREDICTION_FILE = resource_filename('resources.tests', 'mock_predictions.txt')
TEST_DATA_WITH_AIDS = resource_filename('resources.tests', 'test.aid.csv')
TEST_DATA_GT = resource_filename('resources.tests', 'test.wea.csv')
EXPECTED_ACCURACY_AT_5 = {
    "ndcg@5": 0.9081918011004686,
    "num_instances": 600.0,
    "num_questions": 30.0,
    'num_queries_predicted': 30,
    "num_top_answers_correct": 29.0,
    'num_queries_skipped_in_preds_file': 0,
    "average_precision_5_truncated": 1.0,
    "recall@5": 0.8466594516594517,
    "top-1-accuracy": 0.9666666666666667
}

EXPECTED_ACCURACY_AT_10 = {
    "ndcg@10": 0.9256538079729757,
    "num_instances": 600.0,
    "num_questions": 30.0,
    'num_queries_predicted': 30,
    "num_top_answers_correct": 29.0,
    'num_queries_skipped_in_preds_file': 0,
    "average_precision_10_truncated": 0.9650604686318972,
    "recall@10": 0.9758189033189033,
    "top-1-accuracy": 0.9666666666666667
}


class TestComputeRankingStats(TestCase):
    def setUp(self):
        self.given = self
        self.then = self
        self.when = self
        self._and = self

        self.prediction_file, self.ground_truth_file, self.k = None, None, None

    def test_compute_stats_using_relevance_file(self):
        self.given.a_prediction_file()
        self._and.a_ground_truth_file(TEST_DATA_GT)
        self.when.i_compute_stats()
        self.then.predictions_accuracy_matches_expected(EXPECTED_ACCURACY_AT_5)

    def test_compute_stats_using_relevance_file_with_non_default_k(self):
        self.given.a_prediction_file()
        self._and.a_ground_truth_file(TEST_DATA_GT)
        self._and.a_k_set_to(10)
        self.when.i_compute_stats()
        self.then.predictions_accuracy_matches_expected(EXPECTED_ACCURACY_AT_10)

    def a_prediction_file(self):
        self.prediction_file = MOCK_PREDICTION_FILE

    def a_ground_truth_file(self, gt_file):
        self.ground_truth_file = gt_file

    def a_k_set_to(self, k):
        self.k = k

    def i_compute_stats(self):
        with smart_file_open(self.ground_truth_file) as infile:
            ground_truth = RankerRelevanceFileQueryStream(infile)
            with smart_file_open(self.prediction_file) as prediction_file:
                predictions = PredictionReader(prediction_file, file_has_confidence_scores=True)
                if self.k is not None:
                    self.actual_stats, self.top_one_answers = compute_performance_stats(
                        ground_truth_query_stream=ground_truth,
                        prediction_reader=predictions, k=self.k)
                else:
                    self.actual_stats, self.top_one_answers = compute_performance_stats(
                        ground_truth_query_stream=ground_truth,
                        prediction_reader=predictions)

    def predictions_accuracy_matches_expected(self, expected_stats):
        if self.k is None:
            k = 5
        else:
            k = self.k

        for stat_type in ['average_precision_%d_truncated' % k, 'top-1-accuracy', 'num_questions', 'ndcg@%d' % k,
                          'num_instances', 'num_queries_predicted', 'num_queries_skipped_in_preds_file',
                          "recall@%d" % k]:
            self.assertAlmostEqual(expected_stats[stat_type], self.actual_stats[stat_type], delta=EPSILON)
