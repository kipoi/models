from kipoi.model import BaseModel
import numpy as np
from sklearn.externals import joblib
import json
import pandas as pd
import os
import sys
from pathos.multiprocessing import ProcessPool

# access the absolute path to this script
# https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory-in-python
import inspect
this_file_path = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
this_dir = os.path.dirname(this_file_path)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def onehot(seq):
    bases = ['A', 'C', 'G', 'T']
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        if char != "N":
            X[i, bases.index(char.upper())] = 1
    return X


def elongate_intron(intron):
    """
    Elongate intron to be able to predict branchpoints with labranchor. 
    Elongation is with T nucleotide after all donor features up to 94bp.
    """
    insertion = (94 - len(intron)) * "T"
    return ''.join([intron[0:19], insertion, intron[19:]])


def work_seq_on_feature(seqA, seqB, seqD, region, pos, nucl):
    """
    Calculate the feature on the given sequence.
    :return: feature value and sequence index in the soi batch.
    """
    # indexes of start sites in corresponding regions
    B_i = 15
    A_i = 4
    D_i = 3

    value = 0

    if region == 'seqA' and seqA[A_i + int(pos)].upper() == nucl:
        value = 1
    elif region == 'seqB' and seqB[B_i + int(pos)].upper() == nucl:
        value = 1
    elif region == 'seqD' and seqD[D_i + int(pos)].upper() == nucl:
        value = 1

    # TODO _ error
    return (soi_idx, value)


class CleavageTimeModel(BaseModel):

    def __init__(self):
        self.acc_model = joblib.load(os.path.join(this_dir, "model_files/acceptor.pkl"))
        self.don_model = joblib.load(os.path.join(this_dir, "model_files/donor.pkl"))
        self.features_metadata = read_json(os.path.join(this_dir, "model_files/features.json"))
        # acceptor and donor site indexes are unified across SOI
        # NB! This indexes are pos=1 of the region, and index-1 is already pos=-1, not 0!
        self.don_i = 3
        self.acc_i = -21
        # add current dir to python path for multiprocessing
        sys.path.append(this_dir)

    def predict_on_batch(self, x):
        # run feature collection pipeline for the batch
        soi = x

        for i in range(len(soi)):
            if len(soi[i]) < 80:
                soi[i] = elongate_intron(soi[i])

        parameters_batch = self._construct_features_array(soi)

        don_cleavage_time = self.don_model.predict(parameters_batch)
        acc_cleavage_time = self.acc_model.predict(parameters_batch)

        cleavage_time = {'acc_cleavage_time': np.exp(acc_cleavage_time),
                         'don_cleavage_time': np.exp(don_cleavage_time)}

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
        seqA = [seq[self.acc_i - 4: self.acc_i + 6] for seq in soi]
        seqB = [soi[j][int(self.bp_indexes[j]) - 15: int(self.bp_indexes[j]) + 6] for j in range(len(soi))]
        seqD = [seq[self.don_i - 3: self.acc_i + 16] for seq in soi]

        # fill out the rest of the features (base-by-region features)
        for i in range(2, len(self.features_metadata)):
            # parse the current feature info
            (region, pos, nucl) = self.features_metadata[i]
            if (region == 'seqD' or region == 'seqA') and pos > 0:  # decrement, since acc_i/don_i is pos = 1
                pos -= 1

        #    for j in range( len(soi) ):
        #        if region == 'seqB':
        #            i_oi = int(self.bp_indexes[j]) + int(pos)
        #            if soi[j][i_oi].upper() == nucl:
        #                batch_encoded_features[j, i] = 1
        #        else:
        #            if region == 'seqA' and soi[j][ (self.acc_i + int(pos)) ].upper() == nucl:
        #                batch_encoded_features[j, i] = 1
        #            elif region == 'seqD' and soi[j][ (self.don_i + int(pos)) ].upper() == nucl:
        #                batch_encoded_features[j, i] = 1
       #   executor = concurrent.futures.ProcessPoolExecutor(10)
       #    futures = [executor.submit(work_seq_on_feature, seqA[j], seqB[j], seqD[j], region, pos, nucl, j) for j in range(len(soi))]
       #    concurrent.futures.wait(futures)

            pool = ProcessPool(nodes=10)
            feat_column = np.array(pool.map(work_seq_on_feature, seqA, seqB, seqD,
                                            [region for i in range(len(soi))],
                                            [pos for i in range(len(soi))],
                                            [nucl for i in range(len(soi))]))

            # for future in futures:
            #    (seq_idx, value) = future.result()
            #    if value != 0:
            #        feat_column[seq_idx] = value
            batch_encoded_features[:, i] = feat_column

        return batch_encoded_features

    def _get_bp_indexes_labranchor(self, soi):
        """
        Get indexes of branch point regions in given sequences.

        :param soi: batch of sequences of interest for introns (intron-3..intron+6)
        :return: array of predicted bp indexes
        """
        import kipoi
        # prepare labranchor input
        labranchor = kipoi.get_model('labranchor', with_dataloader=False)
        labr_in = np.stack([onehot(seq[self.acc_i - 70:self.acc_i]) for seq in soi], axis=0)
        out = labranchor.predict_on_batch(labr_in)
        # for each row, pick the base with max branchpoint probability, and get its index
        max_indexes = np.apply_along_axis(lambda x: self.acc_i - 70 + np.argmax(x), axis=1, arr=out)

        return max_indexes

    def _count_gc_content(self, seq):
        import collections
        count_gc = collections.Counter(seq)
        return (count_gc['g'] + count_gc['G'] + count_gc['c'] + count_gc['C']) / len(seq)
