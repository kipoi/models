from kipoi.model import BaseModel
import numpy as np
from mmsplice import LOGISTIC_MODEL
from mmsplice.utils.postproc import transform


class MMSpliceModel(BaseModel):
    '''Model to predict delta logit PSI'''

    def predict_on_batch(self, inputs):
        '''inputs shape (,10), corresponding to 5 module predictions of mut and wt'''
        X_alt, X_ref = inputs[:, 5:], inputs[:, :-5]
        X = transform(X_alt - X_ref, True)
        X = np.concatenate([X_ref, X_alt, X[:, -3:]], axis=-1)
        return LOGISTIC_MODEL.predict_proba(X)
