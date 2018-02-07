"""Basenji model
"""

import tensorflow as tf
import numpy as np
import basenji


class Basenji(object):
    """Custom
    """

    def __init__(self, seq_len=262144, model_size="med"):
        """

        Args:
           seq_len: Input sequence length?
        """
        # import pdb; pdb.set_trace()
        
        self.seqnn_model = basenji.seqnn.SeqNN()
        # self.model = SeqNN()
        self.params_file = "model_files/params_{0}.txt".format(model_size)

        job = basenji.dna_io.read_job_params(self.params_file)
        job['batch_length'] = seq_len
        job['seq_depth'] = 4

        self.seq_len = seq_len
        
        # TODO - why is the output 1, 0 , 39?

        self.sess = tf.Session()
        self.seqnn_model.build(job)
        self.seqnn_model.is_training = False
        


        # self.num_targets = ... # TODO

        # hardcoded values
        self.rc = False
        self.shifts = [0]
        self.mc_n = 0
        self.target_indexes = None
        self.return_var  =  False
        self.return_all  =  False
        self.down_sample = 1


    def predict_on_batch(self, x):
        num_seqs = x.shape[0]
        preds_length = x.shape[1]
        # preds = np.zeros((num_seqs, preds_length, self.num_targets), dtype='float16')

        # uniformly sample indexes
        ds_indexes = None
        if self.down_sample != 1:
            ds_indexes = np.arange(0, preds_length, self.down_sample)
            preds_length = len(ds_indexes)

        ensemble_fwdrc = []
        ensemble_shifts = []
        for shift in self.shifts:
            ensemble_fwdrc.append(True)
            ensemble_shifts.append(shift)
        if self.rc:
            ensemble_fwdrc.append(False)
            ensemble_shifts.append(shift)

        # feed dictionary

        fd = self.seqnn_model.set_mode('test')

        preds_batch, preds_batch_var, preds_all = self.seqnn_model._predict_ensemble(
            self.sess,
            fd,
            x,
            ensemble_fwdrc,
            ensemble_shifts,
            self.mc_n,
            ds_indexes,
            self.target_indexes,
            self.return_var,
            self.return_all)
        return preds_batch
