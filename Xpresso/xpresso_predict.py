import sys, gzip
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from mimetypes import guess_type
from optparse import OptionParser
from keras.models import load_model
from Bio import SeqIO
from functools import partial

###################################################################
### Written by Vikram Agarwal (vagar@calicolabs.com)            ###
### Agarwal and Shendure 2020. (https://doi.org/10.1101/416685) ###
### Released free for use under the MIT License                 ###
### Note: These predictions don't include the impact of         ###
###       mRNA half-life as presented in the paper              ###
###################################################################

def main():
    usage = 'usage: %prog [options] <trained_model> <input_file> <output_file>'
    parser = OptionParser(usage)
    parser.add_option('--revCom', dest='revcom', default=False, action='store_true', help='Make predictions for minus strand instead of plus? % [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        print(args)
        parser.error('Must provide mode pre-trained model, input fasta file, and output file name')
    else:
        trained_model = args[0]
        input_file = args[1]
        output_file = args[2]

    def revCom(x):
        for y in range(0,x.shape[0]):
            x[y] = np.fliplr(np.flipud(x[y]))
        return x

    model = load_model(trained_model)

    encoding = guess_type(input_file)[1]  # uses file extension to guess zipped or unzipped
    if encoding is None:
        _open = open
    elif encoding == 'gzip':
        _open = partial(gzip.open, mode='rt')
    else:
        raise ValueError('Unknown file encoding: "{}"'.format(encoding))

    i, bs, names, predictions, sequences = 0, 2, [], [], [] #reduce batch size bs if takes too much memory
    halflifedata = np.zeros((bs,6), dtype='float32')

    with _open(input_file) as f:
        for fasta in SeqIO.parse(f, 'fasta'):
            name, sequence = fasta.id, str(fasta.seq)
            sequences.append(sequence)
            names.append(name)
            i += 1
            if (len(sequence) != 10500):
                sys.exit( "Error in sequence %s, length is not equal to the required 10,500 nts. Please fix or pad with Ns if necessary." % name )
            if i % bs == 0:
                seq = one_hot(sequences)                            
                if options.revcom:
                    seq = revCom(seq)
                predictions.extend( model.predict([seq, halflifedata], batch_size=bs).flatten().tolist() )
                sequences = []

        remain = i % bs
        if remain > 0:
            halflifedata = np.zeros((remain,6), dtype='float32')
            seq = one_hot(sequences)
            if options.revcom:
                seq = revCom(seq)
            predictions.extend( model.predict([seq, halflifedata], batch_size=remain).flatten().tolist() )

        df = pd.DataFrame(np.column_stack((names, predictions)), columns=['ID','SCORE'])
        df.to_csv(output_file, index=False, header=True, sep='\t')

def one_hot(seq):
    num_seqs = len(seq)
    seq_len = len(seq[0])
    seqindex = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
    seq_vec = np.zeros((num_seqs,seq_len,4), dtype='bool')
    for i in range(num_seqs):
        thisseq = seq[i]
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    return seq_vec

if __name__ == '__main__':
    main()