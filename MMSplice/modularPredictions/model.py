from kipoi.model import BaseModel
import sklearn
import numpy as np

# Directly give the modular predictions
class MMSpliceModel(BaseModel):
    
    def predict_on_batch(self, inputs):
    	return inputs