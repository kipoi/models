import numpy as np
from mmsplice import MMSplice
from mmsplice.vcf_dataloader import SplicingVCFDataloader as BaseSplicingVCFDataloader

model = MMSplice(
    exon_cut_l=0,
    exon_cut_r=0,
    acceptor_intron_cut=6,
    donor_intron_cut=6,
    acceptor_intron_len=50,
    acceptor_exon_len=3,
    donor_exon_len=5,
    donor_intron_len=13)


class SplicingVCFDataloader(BaseSplicingVCFDataloader):
    def __next__(self):
        super_out = super().__next__()
        return {
            'inputs': np.concatenate([
                model.predict(super_out['inputs_mut']).values,
                model.predict(super_out['inputs']).values
            ]),
            'metadata': super_out['metadata']
        }
