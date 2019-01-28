from kipoi.model import BaseModel
from mmsplice import LINEAR_MODEL
from mmsplice.utils.postproc import transform


class MMSpliceModel(BaseModel):
    '''Model to predict delta logit PSI'''

    def predict_on_batch(self, inputs):
        return LINEAR_MODEL.predict(transform(inputs, False))
