from kipoi.model import BaseModel
import numpy as np


class HALModel(BaseModel):

    def __init__(self, pkl_file="model_files/HAL_mer_scores.npz", mer_len=6):
        """

        """
        self._bases = ['A', 'T', 'C', 'G']
        self.mer_len = mer_len
        self.model_file = pkl_file
        self.model = self._load_model()

    def _load_model(self):
        w = np.load(self.model_file)['weights']
        self.mer6_dict = dict(zip(self.make_mer_list(6), range(4**6)))
        self.w_mat = w

    def make_mer_list(self, mer_len=6):
        """Makes a list of all n-mers"""
        li = self._bases
        for i in range(self.mer_len - 1):
            li = self._add_base(li)
        return li

    def _add_base(self, li):
        """Used in make_mer_list to add one more base to list"""
        new_li = []
        for s in li:
            for b in self._bases:
                new_li.append(s + b)
        return new_li

    def expit(self, x):
        return 1. / (1. + np.exp(-x))

    def _score_seq_pos(self, seq):
        seq_scores = np.zeros(160)
        for b in range(40):
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 0] / 6.
        for b in range(40, 70):
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 1] / 6.
        for b in [77]:
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 2] / 6.
        for b in [78]:
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 3] / 6.
        for b in [79]:
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 4] / 6.
        for b in [80]:
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 5] / 6.
        for b in range(86, 120 - 5):
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 6] / 6.
        for b in range(115, 155):
            seq_scores[b:b + 6] += self.w_mat[self.mer6_dict[seq[b:b + 6]], 7] / 6.
        return seq_scores

    def predict_seq(self, seq):
        sd_score = sum(self._score_seq_pos(seq))  # 80*2 is hard coded as Rosenberg dose, but can be flexible
        return sd_score

    def _get_x(self, inputs):
        """ Get x for prediction"""
        seq = inputs["inputs"]
        # seq = inputs["seq"]
        return seq

    def predict_on_batch(self, inputs):
        x = self._get_x(inputs)
        if isinstance(x[0], np.ndarray):
            scores = np.array(list(map(self.predict_seq, x[:,1]))) - np.array(list(map(self.predict_seq, x[:,0])))
            return np.array(list(map(self.expit, scores)))
        else:
            scores = list(map(self.predict_seq, x))
            return np.array(scores)
