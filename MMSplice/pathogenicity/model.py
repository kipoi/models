from kipoi.model import BaseModel
import sklearn
import numpy as np
from mmsplice import LOGISTIC_MODEL
from mmsplice.utils.postproc import transform


# Model to predict delta logit PSI
class MMSpliceModel(BaseModel):
    
    def __init__(self):

        self.model = LOGISTIC_MODEL
    
    def predict_on_batch(self, inputs):
        # inputs shape (,10), corresponding to 5 module predictions of mut and wt
        X = inputs[:,:5] - inputs[:,-5:]
        X = transform(X, True)
        X = np.concatenate([inputs, X[:,-3:]], axis=-1)
        pred = self.model.predict_proba(X)
        return pred