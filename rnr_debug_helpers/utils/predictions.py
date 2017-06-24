class Prediction:
    qid = None
    aid = None
    rank_score = None
    conf_score = None

    def __init__(self, rank_score, conf_score, qid=None, aid=None):
        self.qid = qid
        self.aid = aid
        self.rank_score = rank_score
        self.conf_score = conf_score

    def __eq__(self, other):
        is_equal = True
        if not isinstance(other, self.__class__):
            is_equal = False
        elif self.qid != other.qid:
            is_equal = False
        elif self.aid != other.aid:
            is_equal = False
        elif self.rank_score != other.rank_score:
            is_equal = False
        elif self.conf_score is None and other.conf_score is not None:
            is_equal = False
        elif self.conf_score != other.conf_score:
            is_equal = False

        return is_equal

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Prediction(Qid: {}, aid: {}, rank_score: {}, conf_score: {})".format(
            self.qid, self.aid, self.rank_score, self.conf_score)
