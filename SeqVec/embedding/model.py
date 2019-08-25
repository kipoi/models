#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mheinzinger
"""

import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder

from kipoi.model import BaseModel


class Seqvec(BaseModel):
    def __init__(self, weights, options):
        self.seqvec = self.get_model(weights, options)

    def get_model(self, weights, options):
        # Retrieve pre-trained embedding model
        # use GPU if available. If CPU-usage shall be enforced set cuda_device=-1
        cuda_device = 0 if torch.cuda.is_available() else -1
        return ElmoEmbedder(weight_file=weights, options_file=options, cuda_device=cuda_device)

    def predict_single_sample(self, seq):
        return self.seqvec.embed_sentence(list(seq))

    def predict_on_batch(self, inputs, max_chars=15000, verbose=False):
        '''
            Create batches of up to max_chars characters/residues. This speeds
            up the inference at the cost of increased (GPU) memory consumption.
            In case of memory error, you might want to lower this value.
            This value works well for GPUs with 8GB of VRAM.
        '''

        # create containers for storing batches etc
        length_counter = 0
        batch = list()
        emb_dict = list()

        # retrieve embeddings for all protein sequence in the fasta file
        for index, sequence in enumerate(inputs):  # for each protein
            sequence = sequence[0]  # unpacking necessary as strings had to be masked as np.arrays
            # append proteins to batch until max_chars of amino acids are reached
            batch.append(sequence)
            length_counter += len(sequence)

            # Transform list of batches to embeddings
            # if a) max. number of chars. for a batch is reached,
            # if b) sequence is longer than half  max_chars (avoids runtimeError for very long seqs.)
            # if c) the last sequence is reached
            if length_counter > max_chars or len(sequence) > max_chars / 2 or index == len(inputs) - 1:
                if verbose:
                    print('.')
                tokens = [list(seq) for seq in batch]  # necessary pre-processing
                embeddings = self.seqvec.embed_sentences(tokens)  # create generator

                runtime_error = False  # try to retrieve batches if no runtime error occurs
                for batch_idx, seq in enumerate(batch):
                    try:
                        embedding = next(embeddings)  # retrieve embedding from generator
                    except RuntimeError:
                        if verbose:
                            print('RuntimeError for seq_len:({}).'.format(len(seq)))
                            print('Starting single sequence processing')
                        runtime_error = True
                        break
                    # if protein was embedded successfully --> save embedding
                    embedding = self.process_embedding(embedding)
                    emb_dict.append(embedding)

                # Single sequence processing in case of runtime error due to
                # a) very long sequence or b) too large batch size
                # If this fails, you might want to consider lowering max_chars and/or
                # cutting very long sequences into smaller chunks
                if runtime_error:
                    for batch_idx, seq in enumerate(batch):
                        try:  # try single sequence processing
                            embedding = self.seqvec.embed_sentence(tokens[batch_idx])
                        except RuntimeError:
                            if verbose:
                                print('Single sequence processing also not possible. '
                                      'Consider splitting the sequence into smaller seqs.')
                            continue
                        embedding = self.process_embedding(embedding)
                        emb_dict.append(embedding)

                # reset batch-list and character counter
                batch = list()
                length_counter = 0

        emb_dict = np.array(emb_dict)
        return emb_dict

    def process_embedding(self, embedding,
                          residue_reduction=True, protein_reduction=False):
        '''
            Direct output of ELMo has shape (3,L,1024), with L being the protein's
            length, 3 being the number of layers used to train SeqVec (1 CharCNN, 2 LSTMs)
            and 1024 being a hyperparameter chosen to describe each amino acid.
            When a representation on residue level is required, you can sum
            over the first dimension, resulting in a tensor of size (L,1024).
            If you want to reduce each protein to a fixed-size vector, regardless of its
            length, you can average over dimension L.
        '''
        embedding = torch.tensor(embedding)
        if residue_reduction:
            embedding = embedding.sum(dim=0)
        elif protein_reduction:
            embedding = embedding.sum(dim=0).mean(dim=0)

        return embedding.cpu().detach().numpy()
