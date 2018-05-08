from kipoi.model import BaseModel, KerasModel
import numpy as np

class EnsembleModel(BaseModel):
    def __init__(self):
        self.models = [KerasModel(weights="model_files/weights{}.hdf5".format(i),
                                  arch="model_files/arch{}.json".format(i),
                                  backend='tensorflow')
                       for i in range(8)]
        
    def predict_on_batch(self, x):
        # Ensemble model predictions
        return np.stack([m.predict_on_batch(x) for m in self.models]).mean(axis=0)
