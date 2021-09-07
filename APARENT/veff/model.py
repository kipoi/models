from kipoi.model import BaseModel
from keras.models import load_model
import numpy as np

from scipy.special import logit


class APARENTModel(BaseModel):

    def __init__(self, weights, return_size_probabilites=False):
        self.weights = weights
        self.model = load_model(weights)
        self.return_site_probabilities = return_size_probabilites

    def _predict(self, inputs):
        batch_size = inputs.shape[0]

        input_1 = np.expand_dims(inputs, -1)
        input_2 = np.zeros([batch_size, 13])
        input_3 = np.ones([batch_size, 1])

        _, pred = self.model.predict_on_batch([input_1, input_2, input_3])

        site_probs = pred[:, 1:]
        polya_prob = pred[:, 0]
        return site_probs, polya_prob

    def predict_on_batch(self, inputs, return_size_probabilites=False):
        site_probs_ref, polya_prob_ref = self._predict(inputs["ref_seq"])
        site_probs_alt, polya_prob_alt = self._predict(inputs["alt_seq"])

        polya_prob_ref = logit(polya_prob_ref)
        polya_prob_alt = logit(polya_prob_alt)
        site_probs_ref = logit(site_probs_ref)
        site_probs_alt = logit(site_probs_alt)

        return {
            "delta_logit_polya_prob": polya_prob_alt - polya_prob_ref,
            "delta_logit_site_probs": site_probs_alt - site_probs_ref,
        }
