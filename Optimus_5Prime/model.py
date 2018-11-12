from kipoi.model import BaseModel
from keras.models import load_model


class PostProcKerasmodel(BaseModel):

    def __init__(self, weights, postproc_mean, postproc_sd):
        """Simple keras model that also runs the postprocessin
        """
        self.weights = weights
        self.model = load_model(weights)
        self.postproc_mean = postproc_mean
        self.postproc_sd = postproc_sd

    def predict_on_batch(self, inputs):
        out = self.model.predict_on_batch(inputs)
        # run post-processing (inverse of standard-scaler)
        return out * self.postproc_sd + self.postproc_mean