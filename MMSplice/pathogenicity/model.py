from kipoi.model import BaseModel
from mmsplice import MMSplice
from mmsplice.utils import predict_pathogenicity


mmsplice = MMSplice()


class MMSpliceModel(BaseModel):
    '''Model to predict delta logit PSI'''

    def predict_on_batch(self, inputs):
        X_ref = mmsplice.predict_on_batch(inputs['seq'])
        X_alt = mmsplice.predict_on_batch(inputs['mut_seq'])
        return predict_pathogenicity(X_ref, X_alt)
