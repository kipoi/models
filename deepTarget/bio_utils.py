'''
Copyright (C) 2016 Byunghan Lee, Junghwan Baek, Seunghyun Park, and Sungroh Yoon

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
from Bio import SeqIO
import re  # for candidate lists
import itertools
import numpy as np

from keras.preprocessing import sequence
from sklearn.cross_validation import KFold


# definitions for candiate lists
SEED_START = 1
SEED_SIZE = 6  # 6-mer seed
CTS_SIZE = 40

wc_pairs = {"A": "U", "U": "A", "G": "C", "C": "G"}


embd_arch = 'model_files/deepTarget_embd_model.json'
embd_weights = 'model_files/deepTarget_weights_embd_model.hdf5'
arch = 'model_files/deepTarget_model.json'
weights = 'model_files/deepTarget_weights_model.hdf5'


def find_candidate(mirna_sequence, mrna_sequence):
    positions = []

    rev_sequence = mirna_sequence[::-1]
    seed = rev_sequence[-SEED_START - SEED_SIZE:-SEED_START]
    rc_seed = ""

    for i in seed:
        rc_seed += wc_pairs[i]

    it = re.finditer(rc_seed, mrna_sequence)

    for match in it:
        positions.append(match.span()[1] + 1)

    return positions


def get_candidate(mirna_sequence, mrna_sequence, cts_size):
    positions = find_candidate(mirna_sequence, mrna_sequence)

    candidates = []
    for i in positions:
        site_sequence = mrna_sequence[max(0, i - cts_size):i]
        candidates.append(site_sequence)

    return (candidates, positions)


def make_pair(mirna_sequence, mrna_sequence, cts_size=CTS_SIZE):
    candidates, positions = get_candidate(mirna_sequence, mrna_sequence, cts_size)

    mirna_querys = []
    mrna_targets = []
    if len(candidates) == 0:
        return (mirna_querys, mrna_targets)
    else:
        for i in range(len(candidates)):
            mirna_querys.append(mirna_sequence)
            mrna_targets.append(candidates[i])

    return (mirna_querys, mrna_targets)


# functions to deal with fasta file
def read_fasta(mirna_fasta_file, mrna_fasta_file):
    handle = open(mirna_fasta_file, "rU")
    mirna_list = list(SeqIO.parse(handle, "fasta"))
    handle.close()

    handle = open(mrna_fasta_file, "rU")
    mrna_list = list(SeqIO.parse(handle, "fasta"))
    handle.close()

    mirna_ids = []
    mirna_sequences = []
    mrna_ids = []
    mrna_sequences = []

    for i in range(len(mirna_list)):
        mirna_ids.append(str(mirna_list[i].id))
        mirna_sequences.append(str(mirna_list[i].seq))

    for i in range(len(mrna_list)):
        mrna_ids.append(str(mrna_list[i].id))
        mrna_sequences.append(str(mrna_list[i].seq))

    return (mirna_ids, mirna_sequences, mrna_ids, mrna_sequences)

def one_hot(ints):
    dictionary_k = 4  # maximum number of nucleotides
    ints_len = len(ints)
    ints_enc = np.zeros((ints_len, dictionary_k))
    ints_enc[np.arange(ints_len), [k - 1 for k in ints]] = 1

    return ints_enc

def nucleotide_to_int(nucleotide, max_len):
    dictionary = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}

    chars = []
    nucleotide = nucleotide.upper()
    for c in nucleotide:
        chars.append(c)

    ints_enc = [0 for i in range(max_len)]  # for post-padding
    for i in range(len(chars)):
        ints_enc[i] = dictionary.get(chars[i], chars[i])

    #ints_enc = one_hot(ints)
    #ints_enc = ints_enc.tolist()

    return (ints_enc)

def sequence_to_int(sequences, max_len):
    if type(sequences) is list:
        seqs_enc = np.asarray([nucleotide_to_int(read, max_len) for read in sequences], 'uint8')
    else:
        seqs_enc = np.asarray([nucleotide_to_int(read, max_len) for read in sequences], 'uint8')
        seqs_enc = list(itertools.chain(*seqs_enc))
        seqs_enc = np.asarray(seqs_enc)

    return seqs_enc

def remove_post_padding(signal):
    X_not_padded = len(signal) * [[]]

    for i in range(len(signal)):
        pad_initIdx = np.where(signal[i] == 0)
        if np.size(pad_initIdx) == 0:
            X_not_padded[i] = signal[i]
        else:
            X_not_padded[i] = signal[i][0:pad_initIdx[0][0]]

    return (X_not_padded)

def make_pre_padding(X_not_padded, nb_timesteps):
    X_padded = sequence.pad_sequences(X_not_padded, maxlen=nb_timesteps)

    return (X_padded)

def formatting_data(X_query_seqs, X_target_seqs):
    max_len = len(max(X_query_seqs, key=len))
    X_mirna = sequence_to_int(X_query_seqs, max_len)
    max_len = len(max(X_target_seqs, key=len))
    X_mrna = sequence_to_int(X_target_seqs, max_len)

    nb_timesteps = max(X_mirna.shape[1], X_mrna.shape[1])
    X_mirna = remove_post_padding(X_mirna)
    X_mrna = remove_post_padding(X_mrna)
    X_mirna = make_pre_padding(X_mirna, nb_timesteps)
    X_mrna = make_pre_padding(X_mrna, nb_timesteps)

    return (X_mirna, X_mrna)

# functions for cross validation
def split_dataset(nb_samples, nb_folds):
    indicies = KFold(n=nb_samples, n_folds=nb_folds, shuffle=True)
    indicies = list(indicies)  # list in the order of (train, test)

    return (indicies)

# functions to get statistics from a confusion matrix
def get_statistics(confusion_mat):
    confusion_mat = np.double(confusion_mat)

    accuracy = (confusion_mat[0][0] + confusion_mat[1][1]) / (confusion_mat[0][0] + confusion_mat[0][1] + confusion_mat[1][0] + confusion_mat[1][1])
    sensitivity = (confusion_mat[0][0]) / (confusion_mat[0][0] + confusion_mat[1][0])
    specificity = (confusion_mat[1][1]) / (confusion_mat[0][1] + confusion_mat[1][1])
    f1_score = (2 * confusion_mat[0][0]) / (2 * confusion_mat[0][0] + confusion_mat[0][1] + confusion_mat[1][0])
    ppv = (confusion_mat[0][0]) / (confusion_mat[0][0] + confusion_mat[0][1])
    npv = (confusion_mat[1][1]) / (confusion_mat[1][1] + confusion_mat[1][0])

    return (sensitivity, specificity, f1_score, ppv, npv)


def make_train_pair(mirna_ids, mirna_seqs, mrna_ids, mrna_seqs, query_ids, target_ids, labels):
    X_query_ids = []
    X_query_seqs = []
    X_target_ids = []
    X_target_seqs = []
    Y = []

    for i in range(len(query_ids)):
        j = mirna_ids.index(query_ids[i])
        k = mrna_ids.index(target_ids[i])
        query_seqs, target_seqs = make_pair(mirna_seqs[j], mrna_seqs[k])

        nb_pairs = len(query_seqs)
        if nb_pairs > 0:
            querys = [query_ids[i] for k in range(nb_pairs)]
            X_query_ids.extend(querys)
            X_query_seqs.extend(query_seqs)

            targets = [target_ids[i] for k in range(nb_pairs)]
            X_target_ids.extend(targets)
            X_target_seqs.extend(target_seqs)

            y = [labels[i] for k in range(nb_pairs)]
            Y.extend(y)

    Y = np.array([[Y[i]] for i in range(len(Y))], 'uint8')

    return (X_query_ids, X_query_seqs, X_target_ids, X_target_seqs, Y)

def make_test_pair(mirna_ids, mirna_seqs, mrna_ids, mrna_seqs):
    X_query_ids = []
    X_query_seqs = []
    X_target_ids = []
    X_target_seqs = []

    for i in range(len(mirna_ids)):
        for j in range(len(mrna_ids)):
            query_seqs, target_seqs = make_pair(mirna_seqs[i], mrna_seqs[j])

            nb_pairs = len(query_seqs)
            if nb_pairs > 0:
                query_ids = [mirna_ids[i] for k in range(nb_pairs)]
                X_query_ids.extend(query_ids)
                X_query_seqs.extend(query_seqs)

                target_ids = [mrna_ids[j] for k in range(nb_pairs)]
                X_target_ids.extend(target_ids)
                X_target_seqs.extend(target_seqs)

    return (X_query_ids, X_query_seqs, X_target_ids, X_target_seqs)


def make_pair_from_file(mirna_ids, mirna_seqs, mrna_ids, mrna_seqs, query_ids, target_ids):
    X_query_ids = []
    X_query_seqs = []
    X_target_ids = []
    X_target_seqs = []

    for i in range(len(query_ids)):
        j = mirna_ids.index(query_ids[i])
        try:
            k = mrna_ids.index(target_ids[i])
        except ValueError:
            continue

        query_seqs, target_seqs = make_pair(mirna_seqs[j], mrna_seqs[k])

        nb_pairs = len(query_seqs)
        if nb_pairs > 0:
            querys = [query_ids[i] for k in range(nb_pairs)]
            X_query_ids.extend(querys)
            X_query_seqs.extend(query_seqs)

            targets = [target_ids[i] for k in range(nb_pairs)]
            X_target_ids.extend(targets)
            X_target_seqs.extend(target_seqs)

    return (X_query_ids, X_query_seqs, X_target_ids, X_target_seqs)


def read_ground_truth(ground_truth_file):
    query_ids = []
    target_ids = []
    labels = []

    handle = open(ground_truth_file, 'r')
    for line in handle:
        row = line.strip().split('\t')

        query_ids.append(row[0])
        target_ids.append(row[1])
        labels.append(row[2])

    handle.close()

    labels = list(int(labels[l]) for l in range(len(labels)))

    return (query_ids, target_ids, labels)


def read_query_pair(query_pair_file):
    query_ids = []
    target_ids = []

    handle = open(query_pair_file, 'r')
    for line in handle:
        row = line.strip().split('\t')

        query_ids.append(row[0])
        target_ids.append(row[1])

    handle.close()

    return (query_ids, target_ids)
