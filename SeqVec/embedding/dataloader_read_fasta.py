import numpy as np

def read_fasta( fasta_file, split_char=' ', id_field=0 ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file
        In order to retrieve the protein identifier, the header is split 
        after split_char and the field at position id_field is chosen as
        identifier.
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

    sequences = sorted(sequences.items(), key=lambda kv: len( sequences[kv[0]] ) )
    identifier, seqs = zip(*sequences)
    seqs = [ np.asarray([seq]) for seq in seqs ]
    #seqs = np.concatenate( seqs )
    print(seqs)
    return { "inputs": seqs,
             "metadata":
                { "id":  identifier }
            }

