from kipoi.model import KerasModel, BaseModel
import numpy as np

class PrimateDL(BaseModel):
    def __init__(self):
        mdir = "model_files"
        models = [KerasModel(weights=os.path.join(mdir, "0weights{i}.hdf5".format(i=i),
                             arch=os.path.join(mdir, "model.json"),
                             backend='tensorflow')
                  for i in range(8)]

    def predict_on_batch(self, x):
        return np.stack([m.predict_on_batch(x) 
                         for m in self.models]).mean(axis=0)



