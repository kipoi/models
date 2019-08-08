import kipoi

from kipoi.model import BaseModel
# Helper class to plug in SeqVec embeddings for structure prediction
class DeepSeqVec(BaseModel):
    def __init__(self, x=1):
        self.embed  = kipoi.get_model("SeqVec/embedding")
        self.struct = kipoi.get_model("SeqVec/embedding2structure")

    def predict_on_batch(self, x):
        embedding = self.embed.predict_on_batch(x)
        return self.struct.predict_on_batch( embedding ) 
