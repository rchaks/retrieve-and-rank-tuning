"""
@author: rchakravarti
@creation-date: 5/9/17
"""


class LabelledQuery:
    __qid__ = None
    __feature_vectors__ = None
    __labels__ = None
    __rank_score__ = None
    __answer_ids__ = None

    def __init__(self, qid, labels, ans_ids=None, feature_vectors=None, scores=None, conf_scores=None):
        """

        :param str qid: question id or question text
        :param list(int) labels: list of relevance labels
        :param list(str) or None ans_ids: list of document ids (should align with relevance labels). If passed as None,
            then we generate mock ids starting with 0
        :param list(list(float)) or None feature_vectors: optionally include the feature vectors for the answer ids
        :param list(float) or None scores: optionally include the rank scores for the answer ids
        :param list(float) or None conf_scores: optionally include the conf scores for the answer ids
        """
        self.__qid__ = qid
        self.__feature_vectors__ = feature_vectors
        self.__labels__ = labels
        self.__rank_score__ = scores
        self.__conf_score__ = conf_scores
        if ans_ids is not None:
            self.__answer_ids__ = ans_ids
        else:
            self.__answer_ids__ = [x for x in range(len(labels))]

        self._sanity_check_length_of_lists()

    def _sanity_check_length_of_lists(self):
        num_labels = len(self.get_labels())
        if num_labels != len(self.get_answer_ids()):
            raise ValueError('Number of relevance labels (%s) should match the number of provided answer ids (%s): ' %
                             (self.get_labels(), self.get_answer_ids()))

        if self.get_feature_vectors() is not None and len(self.get_feature_vectors()) != num_labels:
            raise ValueError('Number of relevance labels (%s) should match the number of provided feature vecs (%s): ' %
                             (self.get_labels(), self.get_feature_vectors()))

        if self.get_rank_scores() is not None and len(self.get_rank_scores()) != num_labels:
            raise ValueError('Number of relevance labels (%s) should match the number of provided scores (%s): ' %
                             (self.get_labels(), self.get_rank_scores()))

    def add_candidate_answer(self, label, feature_vector=None, rank_score=None, answer_id=None):

        self.__labels__.append(label)

        if answer_id is not None:
            if answer_id in self.__answer_ids__:
                raise ValueError("Trying to add a answer id which already exists: <<%s>>" % answer_id)
            else:
                self.__answer_ids__.append(answer_id)
        else:
            # increment and use next spot
            self.__answer_ids__.append(len(self.__answer_ids__))

        if self.__feature_vectors__ is None:
            self.__feature_vectors__ = [feature_vector]
        else:
            self.__feature_vectors__.append(feature_vector)

        if rank_score is not None:
            if self.__rank_score__ is not None:
                self.__rank_score__.append(rank_score)
            else:
                self.__rank_score__ = [rank_score]

    def get_qid(self):
        return self.__qid__

    def get_answer_ids(self):
        return self.__answer_ids__

    def set_answer_ids(self, answer_ids):
        self.__answer_ids__ = answer_ids

    def get_answer_count(self):
        return len(self.__answer_ids__)

    def get_feature_vectors(self):
        return self.__feature_vectors__

    def set_feature_vector(self, answer_id, feature_vector):
        self.__feature_vectors__[self.__answer_ids__.index(answer_id)] = feature_vector

    def get_feature_vector(self, answer_id):
        return self.__feature_vectors__[self.__answer_ids__.index(answer_id)]

    def get_labels(self):
        return self.__labels__

    def set_label(self, answer_id, label):
        self.__labels__[self.__answer_ids__.index(answer_id)] = label

    def get_label(self, answer_id):
        return self.__labels__[self.__answer_ids__.index(answer_id)]

    def set_labels(self, labels):
        self.__labels__ = labels

    def get_conf_scores(self):
        return self.__conf_score__

    def get_conf_score(self, answer_id):
        if self.__conf_score__:
            return self.__conf_score__[self.__answer_ids__.index(answer_id)]
        return None

    def set_conf_scores(self, scores):
        self.__conf_score__ = scores

    def get_rank_scores(self):
        return self.__rank_score__

    def get_rank_score(self, answer_id):
        if self.__rank_score__:
            return self.__rank_score__[self.__answer_ids__.index(answer_id)]
        return None

    def set_rank_scores(self, scores):
        self.__rank_score__ = scores

    def to_csv_row_for_relevance_file(self):
        row = ['%s' % self.get_qid()]
        for doc_id in self.get_answer_ids():
            if self.get_label(doc_id) > 0:
                row.append('%s' % doc_id)
                row.append('%s' % self.get_label(doc_id))
        return row

    def __str__(self):
        return '%s' % {'query text': self.get_qid(), 'feature vectors': self.get_feature_vectors(),
                       'labels': self.get_labels(),
                       'rank scores': self.get_rank_scores(), 'doc ids': self.get_answer_ids()}
