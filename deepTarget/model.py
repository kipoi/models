import os
os.environ["KERAS_BACKEND"] = "theano"

import sys
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
sys.path.append(this_path)
from kipoi.model import BaseModel
from keras.models import model_from_json
import bio_utils


class Model(BaseModel):
    embd_arch = bio_utils.embd_arch
    embd_weights = bio_utils.embd_weights
    arch = bio_utils.arch
    weights = bio_utils.weights

    def __init__(self):
        self.embd_model = model_from_json(open(self.embd_arch).read())
        self.embd_model.load_weights(self.embd_weights)
        self.model = model_from_json(open(self.arch).read())
        self.model.load_weights(self.weights)

    def predict_on_batch(self, inputs):
        # pre-compute the embeddings
        X_mirna_embd = self.embd_model.predict(inputs["mirna_int_seq"])
        X_mrna_embd = self.embd_model.predict(inputs["mrna_int_seq"])
        return self.model.predict([X_mirna_embd, X_mrna_embd], verbose=0)
