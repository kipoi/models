from kipoi.model import BaseModel
import numpy as np
from maxentpy import maxent
from maxentpy.maxent import load_matrix5, load_matrix3

class MaxEntModel(BaseModel):

    def __init__(self, side='5prime'):
        """

        """
        self.side = side
        matrix5 = load_matrix5()
        matrix3 = load_matrix3()
        if self.side == '5prime':
            self.matrix = matrix5
            self.model = maxent.score5
        else:
            self.matrix = matrix3
            self.model = maxent.score3

    def _score_seq(self, seq):
        seq_score = self.model(seq, matrix=self.matrix)
        return seq_score

    def score_seqs(self, seqs):
        seq_scores = [self._score_seq(seq) for seq in seqs]
        return seq_scores

    def _get_x(self, inputs):
        """ Get x for prediction"""
        #seq = inputs["seq"]
        seq = inputs['inputs']["seq"]
        return seq

    def predict_on_batch(self, inputs):
        x = self._get_x(inputs)
        pred = list(map(self.score_seqs, x))
        return np.array(pred)
