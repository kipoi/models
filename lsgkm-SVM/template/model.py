from __future__ import print_function

from kipoi.model import BaseModel
import numpy as np
import subprocess
import os
import tempfile



def encode_sequence_into_fasta_file(seq, ofname):
    """writes sequences into fasta file
    """
    with open(ofname, "w") as wf:
        for i, seq_here in enumerate(seq):
            print('>{}'.format(i), file=wf)
            print(seq_here, file=wf)


class LsgkmSvm(BaseModel):
    def __init__(self, model_file = "model_files/model.txt", threads=1):
        self.model_file = model_file
        self.threads = threads

    #### from https://github.com/kundajelab/dragonn/blob/master/dragonn/models.py#L336
    def predict_on_batch(self, inputs):
        # write test fasta file
        temp_input = tempfile.NamedTemporaryFile(suffix = ".txt")
        test_fname = temp_input.name
        encode_sequence_into_fasta_file(ofname = test_fname, seq = inputs.tolist())
        # test gkmsvm
        temp_ofp = tempfile.NamedTemporaryFile(suffix = ".txt")
        threads_option = '-T %s' % (str(self.threads))
        verbosity_option = '-v 0'
        command = ' '.join(['gkmpredict',
                            test_fname,
                            self.model_file,
                            temp_ofp.name,
                            threads_option,
                            verbosity_option])
        #process = subprocess.Popen(command, shell=True)
        #process.wait()  # wait for it to finish
        exit_code = os.system(command)
        temp_input.close()
        assert exit_code == 0
        # get classification results
        temp_ofp.seek(0)
        y = np.array([line.split()[-1] for line in temp_ofp], dtype=float)
        temp_ofp.close()
        return np.expand_dims(y, 1)

