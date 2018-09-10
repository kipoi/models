from kipoi.model import BaseModel
import numpy as np
from sklearn.externals import joblib
import json
import pandas as pd
from math import log, exp
import os

# access the absolute path to this script
# https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory-in-python
import inspect
this_file_path = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
this_dir = os.path.dirname(this_file_path)

def load_acc_model():
    return joblib.load("model_files/acceptor.pkl")

def load_don_model():
    return joblib.load("model_files/donor.pkl")
    
def load_features_metadata():
    fh = open("model_files/features.json")
    info = json.load(fh)
    fh.close()
    return info

def onehot(seq):
    bases = ['A', 'C', 'G', 'T']
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        if char != "N":
            X[i, bases.index(char.upper())] = 1
    return X
    
class CleavageTimeModel(BaseModel):
    
    def __init__(self):
        self.don_model = load_don_model()
        self.acc_model = load_acc_model()
        self.features_metadata = load_features_metadata()
        # acceptor and donor site indexes are unified across SOI
        # NB! This indexes are pos=1 of the region, and index-1 is already pos=-1, not 0!
        self.don_i = 3
        self.acc_i = -21
        
    def predict_on_batch(self, x):
        # run feature collection pipeline for the batch
        soi = x["soi"]
        self.bp_indexes = x["bp_indexes"]
        parameters_batch = self._construct_features_array(soi)
        
        don_cleavage_time = self.don_model.predict(parameters_batch)
        acc_cleavage_time = self.acc_model.predict(parameters_batch)
        
        cleavage_time = {'acc_cleavage_time': [exp(tm) for tm in acc_cleavage_time], 'don_cleavage_time': [exp(tm) for tm in don_cleavage_time]}
        
        return cleavage_time
        
        
    def _construct_features_array(self,soi):
        """
        Constructs features array.
        
        :return: numpy array for running the model.
        """
        
        shape = ( len(soi), len(self.features_metadata) )
        batch_encoded_features = np.zeros( shape )
        
        # first feature is the gc content in acceptor region (double acceptor window at the end)
        acceptors = [ x[ 2*self.acc_i :] for x in soi ] 
        batch_encoded_features[:, 0] = np.array( [ self._count_gc_content(acceptor) for acceptor in acceptors ] )
        # second feature is gc content in intron region
        introns = [ x[ self.don_i : self.acc_i ] for x in soi ]
        batch_encoded_features[:, 1] = np.array( [ self._count_gc_content(intron) for intron in introns ] )
        
        # fill out the rest of the features (base-by-region features)
        for i in range(2, len(self.features_metadata)):
            # parse the current feature info
            (region, pos, nucl) = self.features_metadata[i]
            if (region == 'seqD' or region == 'seqA') and pos > 0: #decrement, since acc_i/don_i is pos = 1
                pos -= 1

            for j in range( len(soi) ):
                if len(soi[j]) < 80:
                    #ignore short introns. TODO: behaviour in this case
                    continue
                if region == 'seqB':
                    i_oi = int(self.bp_indexes[j]) + int(pos)
                    if soi[j][i_oi].upper() == nucl:
                        batch_encoded_features[j, i] = 1
                else:
                    if region == 'seqA' and soi[j][ (self.acc_i + int(pos)) ].upper() == nucl:
                        batch_encoded_features[j, i] = 1
                    elif region == 'seqD' and soi[j][ (self.don_i + int(pos)) ].upper() == nucl:
                        batch_encoded_features[j, i] = 1
        return batch_encoded_features
    
    def _count_gc_content(self, seq):
        import collections 
        count_gc = collections.Counter(seq)
        return (count_gc['g'] + count_gc['G'] + count_gc['c'] + count_gc['C'])/len(seq)
        
        
        
        
