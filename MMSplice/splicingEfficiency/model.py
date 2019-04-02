from kipoi.model import BaseModel
from mmsplice import EFFICIENCY_MODEL
from mmsplice.utils.postproc import transform


class MMSpliceModel(BaseModel):
    '''Model to predict delta logit PSI'''

    def predict_on_batch(self, inputs):
        '''inputs shape (,10), corresponding to 5 module predictions of mut and wt'''
        X = transform(inputs[:, :5] - inputs[:, -5:], True)[:, [1, 2, 3, 5]]
        return EFFICIENCY_MODEL.predict(X)
