from kipoi.model import BaseModel
import numpy as np
from mmsplice import LINEAR_MODEL
from mmsplice.utils.postproc import transform

# Model to predict delta logit PSI
class MMSpliceModel(BaseModel):
    
    def __init__(self):

        self.model = LINEAR_MODEL
    
    def predict_on_batch(self, inputs):
        X = transform(inputs, False)
        pred = self.model.predict(X)
        return pred