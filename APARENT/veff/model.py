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

        site_probs = pred[:, :-1]
        distal_prob = pred[:, -1]
        return site_probs, distal_prob

    def predict_on_batch(self, inputs):
        site_probs_ref, distal_prob_ref = self._predict(inputs["ref_seq"])
        site_probs_alt, distal_prob_alt = self._predict(inputs["alt_seq"])

        distal_prob_ref = logit(distal_prob_ref)
        distal_prob_alt = logit(distal_prob_alt)
        proximal_prob_ref = logit(np.sum(
            site_probs_ref[:, self.isoform_window_start:self.isoform_window_end],
            axis=1
        ))
        proximal_prob_alt = logit(np.sum(
            site_probs_alt[:, self.isoform_window_start:self.isoform_window_end],
            axis=1
        ))

        return {
            "delta_logit_distal_prob": distal_prob_alt - distal_prob_ref,
            "delta_logit_proximal_prob": proximal_prob_alt - proximal_prob_ref,
        }
