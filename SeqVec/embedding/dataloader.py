
import numpy as np
from kipoi.data import Dataset

class SeqDataloader(Dataset):
    """
    Args:
        fasta_file: file path; Protein sequence(s)
    """

    def __init__(self, fasta_file, split_char=' ', id_field=0 ):

        seq_dict = self.read_fasta( fasta_file, split_char, id_field )
        self.length = len(seq_dict)
        
        sequences = sorted(seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ) )
        self.identifier, self.seqs = zip(*sequences)
        self.seqs = [ np.asarray([seq]) for seq in self.seqs ]
        
    def read_fasta( self, fasta_file, split_char, id_field ):
        '''
            Reads in fasta file containing multiple sequences.
            Returns dictionary of holding multiple sequences or only single 
            sequence, depending on input file.
        '''
        
        sequences = dict()
        with open( fasta_file, 'r' ) as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                    sequences[ uniprot_id ] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    sequences[ uniprot_id ] += ''.join( line.split() ).upper()
        return sequences

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "inputs": self.seqs[idx],
            "metadata": {
                "id": self.identifier[idx]
            }
        }

