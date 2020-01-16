from kipoi.model import BaseModel
import numpy as np
from mmsplice import MMSplice


mmsplice = MMSplice()


class MMSpliceModel(BaseModel):
    '''Directly give the modular predictions'''

    def predict_on_batch(self, inputs):
        X_ref = mmsplice.predict_on_batch(inputs['seq'])
        X_alt = mmsplice.predict_on_batch(inputs['mut_seq'])
        return np.hstack((X_alt, X_ref))
