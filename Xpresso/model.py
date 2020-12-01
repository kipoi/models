from kipoi.model import BaseModel
from keras.models import load_model
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np

class Xpresso(BaseModel):
        def __init__(self, weights):
            self.nuc_dict = {'A':[1 ,0 ,0 ,0 ],'C':[0 ,1 ,0 ,0 ],'G':[0 ,0 ,1 ,0 ],
                             'T':[0 ,0 ,0 ,1 ],'N':[0 ,0 ,0 ,0 ]}
            self.weights = weights
            self.model = load_model(weights)

        # Predicts for a batch of inputs
        def predict_on_batch(self, inputs):
            #In this limited model, treat RNA as having average mRNA features, to ignore half-life contribution
            #For full model with half-life features, see Xpresso Github
            mean_half_life_features = np.zeros((inputs.shape[0],6), dtype='float32')
            pred = self.model.predict_on_batch([inputs, mean_half_life_features]).reshape(-1)
            return {"expression_pred": pred}
