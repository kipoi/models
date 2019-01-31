from kipoi.model import BaseModel


class MMSpliceModel(BaseModel):
    '''Directly give the modular predictions'''

    def predict_on_batch(self, inputs):
        return inputs
