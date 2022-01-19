from kipoi.model import BaseModel
from keras.models import load_model
import numpy as np

from scipy.special import logit


class APARENTModel(BaseModel):

    def __init__(self, weights, isoform_window_start=0, isoform_window_end=205):
        self.weights = weights
        self.model = load_model(weights)
        self.isoform_window_start = isoform_window_start
        self.isoform_window_end = isoform_window_end

    def _predict(self, inputs):
        batch_size = inputs.shape[0]

        input_1 = np.expand_dims(inputs, -1)
        input_2 = np.zeros([batch_size, 13])
        input_3 = np.ones([batch_size, 1])

        _, pred = self.model.predict_on_batch([input_1, input_2, input_3])

        site_props = pred[:, :-1]
        distal_prop = pred[:, -1]
        return site_props, distal_prop

    def predict_on_batch(self, inputs):
        site_props_ref, distal_prop_ref = self._predict(inputs["ref_seq"])
        site_props_alt, distal_prop_alt = self._predict(inputs["alt_seq"])

        logit_distal_prop_ref = logit(distal_prop_ref)
        logit_distal_prop_alt = logit(distal_prop_alt)

        logit_proximal_prop_ref = logit(np.sum(
            site_props_ref[:, self.isoform_window_start:self.isoform_window_end],
            axis=1
        ))
        logit_proximal_prop_alt = logit(np.sum(
            site_props_alt[:, self.isoform_window_start:self.isoform_window_end],
            axis=1
        ))

        return {
            "delta_logit_distal_prop": logit_distal_prop_alt - logit_distal_prop_ref,
            "delta_logit_proximal_prop": logit_proximal_prop_alt - logit_proximal_prop_ref,
        }
