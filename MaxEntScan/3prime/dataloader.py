import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))


# attach template to pythonpath
import sys
sys.path.append(os.path.join(this_path, "../template"))

from dataloader import SplicingMaxEntDataset


class SplicingMaxEntDatasetSpec(SplicingMaxEntDataset):

    def __init__(self,
                 gtf_file,
                 fasta_file,
                 target_file=None,
                 MISO_AS=False,
                 label_col='event_name'):
        """
        Args:
            gtf_file: gtf file. Can be dowloaded from MISO or ensembl.
            fasta_file: file path; Genome sequence
            target_file: file path; path to the targets in MISO summary format.
            MISO_AS: whether the used annotation file is from MISO alternative splicing annotation.
            label_col: column name in target file which has PSI.
        """
        super(SplicingMaxEntDatasetSpec, self).__init__(
            gtf_file=gtf_file,
            fasta_file=fasta_file,
            side='3prime',
            target_file=target_file,
            MISO_AS=MISO_AS,
            label_col=label_col
        )
