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
import keras.backend as K


class Model(BaseModel):

    def __init__(self, embd_arch, embd_weights, arch, weights):
        self.embd_model = model_from_json(open(embd_arch).read())
        self.embd_model.load_weights(embd_weights)
        self.model = model_from_json(open(arch).read())
        self.model.load_weights(weights)

    def predict_on_batch(self, inputs):
        # pre-compute the embeddings
        X_mirna_embd = self.embd_model.predict(inputs["mirna_int_seq"])
        X_mrna_embd = self.embd_model.predict(inputs["mrna_int_seq"])
        return self.model.predict([X_mirna_embd, X_mrna_embd], verbose=0)
