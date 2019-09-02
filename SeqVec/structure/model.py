import kipoi
from kipoi.model import BaseModel
import os
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_dir = os.path.dirname(os.path.abspath(filename))


# Helper class to plug in SeqVec embeddings for structure prediction
class DeepSeqVec(BaseModel):
    def __init__(self, x=1):
        self.embed  = kipoi.get_model(os.path.join(this_dir, "../embedding"), source='dir')
        self.struct = kipoi.get_model(os.path.join(this_dir, "../embedding2structure"), source='dir')

    def predict_on_batch(self, x):
        embedding = self.embed.predict_on_batch(x)
        return self.struct.predict_on_batch( embedding ) 
