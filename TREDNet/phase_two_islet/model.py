# -*- coding: utf-8 -*-
"""
@author: okurman
"""
import kipoi
from kipoi.model import KerasModel
import inspect
import os
import numpy as np

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_dir = os.path.dirname(os.path.abspath(filename))


class PhaseTwoModel(KerasModel):

    def __init__(self, weights):
        super().__init__(weights=weights)
        self.phase_one_model = kipoi.get_model(os.path.join(this_dir, "../phase_one"), source="dir")

    def predict_on_batch(self, x):

        Y_1 = self.phase_one_model.predict_on_batch(x)
        Y_1 = Y_1[..., np.newaxis]
        Y_2 = self.model.predict_on_batch(Y_1)

        return Y_2