"""
@author: rchakravarti
@creation-date: 12/21/16
"""
import sys
from unittest import TestCase

from rnr_debug_helpers.utils import stats

_EPSILON = 4


class TestStats(TestCase):
    def test_ndcg(self):
        ideal_gt_ordering = [2, 2, 1, 1, 1]
        actual_gt_ordering = [0, 1, 0, 0, 0]
        expected_ndcg = 0.1016
        self.assertAlmostEqual(expected_ndcg, stats.ndcg(actual_gt_ordering, ideal_gt_ordering, k=5), places=_EPSILON)

    def test_ndcg_truncation(self):
        ideal_gt_ordering = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        actual_gt_ordering = [0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
        expected_ndcg = 0.1016
        self.assertAlmostEqual(expected_ndcg, stats.ndcg(actual_gt_ordering, ideal_gt_ordering, k=5), places=_EPSILON)
