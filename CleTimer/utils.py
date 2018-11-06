import json
import numpy as np
import os


def read_json(path):
    with open(path) as fh:
        info = json.load(fh)
    return info


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


# --------------------------------------------
# Old code


# def work_seq_on_feature(seqA, seqB, seqD, region, pos, nucl):
#    """
#    Calculate the feature on the given sequence.
#    :return: feature value and sequence index in the soi batch.
#    """
#    # indexes of start sites in corresponding regions
#    B_i = 15
#    A_i = 4
#    D_i = 3
#
#    value = 0
#
#    if region == 'seqA' and seqA[ A_i + int(pos) ].upper() == nucl:
#        value = 1
#    elif region == 'seqB' and seqB[B_i + int(pos)].upper() == nucl:
#        value = 1
#    elif region == 'seqD' and seqD[ D_i + int(pos) ].upper() == nucl:
#        value = 1
#
#    return (soi_idx, value)
#
