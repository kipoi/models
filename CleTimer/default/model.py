from kipoi.model import BaseModel
import numpy as np
import numpy.core.defchararray as npc
from sklearn.externals import joblib
import json
import pandas as pd
from math import log, exp
import os
import sys
import kipoi

# shared utils
from utils import read_json, onehot, elongate_intron

# access the absolute path to this script
# https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory-in-python
import inspect
this_file_path = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
this_dir = os.path.dirname(this_file_path)


class CleavageTimeModel(BaseModel):

    def __init__(self, acc_model, don_model, features_path=None):
        self.don_model = joblib.load(don_model)
        self.acc_model = joblib.load(acc_model)
        if features_path is None:
            features_path = os.path.join(this_dir, "../features.json")
        self.features_metadata = read_json(features_path)
        # acceptor and donor site indexes are unified across SOI
        # NB! This indexes are pos=1 of the region, and index-1 is already pos=-1, not 0!
        self.don_i = 3
        self.acc_i = -21
        self.labranchor = kipoi.get_model("labranchor", with_dataloader=False)
        # add current dir to python path for multiprocessing
        sys.path.append(this_dir)

    def predict_on_batch(self, x):
        # run feature collection pipeline for the batch
        soi = x.astype(str)  # make sure the type is right

        for i in range(len(soi)):
            if len(soi[i]) < 94:
                soi[i] = elongate_intron(soi[i])

        parameters_batch = self._construct_features_array(soi)

        don_cleavage_time = self.don_model.predict(parameters_batch)
        acc_cleavage_time = self.acc_model.predict(parameters_batch)

        cleavage_time = {'acc_cleavage_time': np.exp(acc_cleavage_time), 'don_cleavage_time': np.exp(don_cleavage_time)}

        return cleavage_time

    def _construct_features_array(self, soi):
        """
        Constructs features array.

        :return: numpy array for running the model.
        """

        shape = (len(soi), len(self.features_metadata))
        batch_encoded_features = np.zeros(shape)

        # first feature is the gc content in acceptor region (double acceptor window at the end)
        acceptors = [x[2 * self.acc_i:] for x in soi]
        batch_encoded_features[:, 0] = np.array([self._count_gc_content(acceptor) for acceptor in acceptors])
        # second feature is gc content in intron region
        introns = [x[self.don_i: self.acc_i] for x in soi]
        batch_encoded_features[:, 1] = np.array([self._count_gc_content(intron) for intron in introns])

        # get the list of bp index for each sequence of batch
        self.bp_indexes = self._get_bp_indexes_labranchor(soi)
        # slice out feature sequences
        # seqA = [ seq[self.acc_i - 4 : self.acc_i + 6] for seq in soi]
        seqB = np.array([soi[j][int(self.bp_indexes[j]) - 15: int(self.bp_indexes[j]) + 6] for j in range(len(soi))])
        B_i = 15
        # seqD = [ seq[self.don_i - 3 : self.acc_i + 16] for seq in soi]

        # fill out the rest of the features (base-by-region features)
        for i in range(2, len(self.features_metadata)):
            # parse the current feature info
            (region, pos, nucl) = self.features_metadata[i]
            if (region == 'seqD' or region == 'seqA'):  # decrement, since acc_i/don_i is pos = 1
                if pos > 0:
                    pos -= 1
                # apply vectorized numpy operations
                if region == 'seqD':
                    idx = self.don_i + int(pos)
                else:
                    idx = self.acc_i + int(pos)
                feat_column = npc.find(soi, nucl, idx, idx + 1)
            else:
                idx = B_i + int(pos)
                feat_column = npc.find(seqB, nucl, idx, idx + 1)

            feat_column[feat_column > 1] = 1
            feat_column[feat_column == -1] = 0
            batch_encoded_features[:, i] = feat_column

        return batch_encoded_features

    def _get_bp_indexes_labranchor(self, soi):
        """
        Get indexes of branch point regions in given sequences.

        :param soi: batch of sequences of interest for introns (intron-3..intron+6)
        :return: array of predicted bp indexes
        """
        encoded = [onehot(str(seq)[self.acc_i - 70:self.acc_i]) for seq in np.nditer(soi)]
        labr_in = np.stack(encoded, axis=0)
        out = self.labranchor.predict_on_batch(labr_in)
        # for each row, pick the base with max branchpoint probability, and get its index
        max_indexes = np.apply_along_axis(lambda x: self.acc_i - 70 + np.argmax(x), axis=1, arr=out)
        # self.write_bp(max_indexes)
        return max_indexes

# TODO boilerplate
#    def write_bp(self, max_indexes):
#        max_indexes = [str(seq) for seq in np.nditer(max_indexes)]
#        with open(''.join([this_dir, "/../customBP/example_files/bp_idx_chr21_labr.txt"]), "a") as bp_idx_file:
#            bp_idx_file.write('\n'.join(max_indexes))
#            bp_idx_file.write('\n')
#            bp_idx_file.close()

    def _count_gc_content(self, seq):
        import collections
        count_gc = collections.Counter(seq)
        return (count_gc['g'] + count_gc['G'] + count_gc['c'] + count_gc['C']) / len(seq)
