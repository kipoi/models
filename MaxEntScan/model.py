from kipoi.model import BaseModel
import numpy as np
from maxentpy.maxent import score5, score3, load_matrix5, load_matrix3
import pdb


class MaxEntModel(BaseModel):

    def __init__(self, side='5prime'):
        """

        """
        if side not in ['5prime', '3prime']:
            raise Exception("side should be 5prime or 3prime")
        self.side = side
        if self.side == '5prime':
            self.matrix = load_matrix5()
            self.model = score5
        else:
            self.matrix = load_matrix3()
            self.model = score3

    def _score_seq(self, seqs):
        """
        Score one sequence pair.
        Calculate the delta maxent score between alternative donors/acceptors
        """
        seq_scores = [self.model(seq, matrix=self.matrix) for seq in seqs]
        if self.side == '5prime':
            seq_score = seq_scores[1] - seq_scores[0]
        else:
            seq_score = seq_scores[0] - seq_scores[1]
        return seq_score

    def score_seqs(self, seqs):
        """ score multiple sequence pairs
        """
        seq_scores = [self._score_seq(seq) for seq in seqs]
        return np.array(seq_scores)

    def _get_x(self, inputs):
        """ Get x for prediction"""
        seq = np.array(inputs["seq"]).T
        # seq = np.array(inputs['inputs']["seq"]).T
        return seq

    def predict_on_batch(self, inputs):
        x = self._get_x(inputs)
        pred = self.score_seqs(x)
        return np.array(pred)
