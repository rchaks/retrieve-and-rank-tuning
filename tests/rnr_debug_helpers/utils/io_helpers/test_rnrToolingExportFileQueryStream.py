# -*- coding: utf-8 -*-
from unittest import TestCase

from pkg_resources import resource_filename

from rnr_debug_helpers.utils.io_helpers import smart_file_open, RnRToolingExportFileQueryStream

_MOCK_RELEVANCE_FILE = resource_filename('resources.tests', 'mock_relevance_file.json')
_EXPECTED_QUERIES = ["my first query", "my second query without double quotes", "my third query with utf-8 char ñ",
                     "my third query with comma,", "my last query with no labelled answers"]
_EXPECTED_DOC_IDS = [["doc_id_1", "doc_id_24", "doc_id_7000"], ["doc_id_36", "doc_id_2", "doc_id_3"],
                     ["doc_id_36", "doc_id_2", "doc_id_3"], ["doc_id_36", "doc_id_2", "doc_id_3"], []]
_EXPECTED_LABELS = [[0, 4, 3], [0, 0, 0], [4, 0, 0], [4, 0, 0], []]


class TestRelevanceFileQueryStream(TestCase):
    def setUp(self):
        self.given = self
        self.then = self
        self.when = self
        self._and = self

    def test_query_stream_from_relevance_file(self):
        self.given.a_relevance_file(_MOCK_RELEVANCE_FILE)
        self.when.i_read_in_labelled_queries()
        self.then.i_expect_num_queries_to_match_expected(len(_EXPECTED_QUERIES))
        self._and.i_expect_the_query_details_to_match_expected(_EXPECTED_QUERIES, _EXPECTED_DOC_IDS, _EXPECTED_LABELS)

    def a_relevance_file(self, relevance_file):
        self.input_relevance_file = relevance_file

    def i_read_in_labelled_queries(self):
        self.queries_read_from_stream = list()
        with smart_file_open(self.input_relevance_file, mode='r') as infile:
            query_stream = RnRToolingExportFileQueryStream(infile)
            for labelled_query in query_stream:
                self.queries_read_from_stream.append(labelled_query)

    def i_expect_num_queries_to_match_expected(self, num_expected):
        self.assertEquals(len(self.queries_read_from_stream), num_expected)

    def i_expect_the_query_details_to_match_expected(self, expected_query_texts, expected_doc_ids, expected_labels):
        i = 0
        for actual_query in self.queries_read_from_stream:
            try:
                self.assertEquals(expected_doc_ids[i], actual_query.get_answer_ids())
                self.assertEquals(expected_query_texts[i], actual_query.get_qid())
                self.assertEquals(expected_labels[i], actual_query.get_labels())
            except AssertionError:
                print('Unexpected values for %d query: %s' % (i + 1, actual_query))
                raise
            i += 1
