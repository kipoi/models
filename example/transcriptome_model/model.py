from kipoi.model import BaseModel
import numpy as np


class MyModel(BaseModel):  # Implement your Kipoi model
    def __init__(self, model_file, some_param):
        """Simple linear model
        """
        self.weights = np.loadtxt(model_file).astype(float)
        self.some_param = some_param

    # Execute model prediction for input data
    def predict_on_batch(self, x):  # The bare minimum that has to be defined
        return x @ self.weights * self.some_param
