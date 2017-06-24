class Answer:
    """
    Class defines a data structure to hold question id, ground truth, rank score, and confidence score for an answer.
    If confidence score isn't provided, just uses the rank score. Implements a natural ordering based on the rank score.
    """

    def __init__(self, qid, ground_truth, score, answer_id=None, confidence=None):
        self.qid = qid
        self.answer_id = answer_id
        self.ground_truth = int(ground_truth)
        self.score = float(score)
        if confidence is not None:
            self.confidence = float(confidence)
        else:
            self.confidence = self.score

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return not self.__eq__(other)
