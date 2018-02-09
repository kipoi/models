import numpy as np
import sys
import os
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))
sys.path.append(this_path)
import bio_utils

# examples
mirna_fasta_file = 'example_files/miRNA.fasta'
mrna_fasta_file = 'example_files/3UTR.fasta'
query_pair_file = 'example_files/miRNA-mRNA_query.txt'
prediction_file = 'example_files/predictions.txt'


def dataset(mirna_fasta_file, mrna_fasta_file, query_pair_file):
    """Load all the data into memory

    Args:
      mirna_fasta_file: mirna fasta file
      mrna_fnasta_file: mrna fasta file of interest
      query_pair_file: ?
    """
    print('\nLoading data: ', mirna_fasta_file, ' and ', mrna_fasta_file, '...')
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = bio_utils.read_fasta(mirna_fasta_file, mrna_fasta_file)
    query_ids, target_ids = bio_utils.read_query_pair(query_pair_file)

    print('\nPreprocessing inputs to deepTarget...')
    X_query_ids, X_query_seqs, X_target_ids, X_target_seqs = bio_utils.make_pair_from_file(mirna_ids,
                                                                                           mirna_seqs,
                                                                                           mrna_ids,
                                                                                           mrna_seqs,
                                                                                           query_ids,
                                                                                           target_ids)

    # seq -> int encoding
    X_mirna, X_mrna = bio_utils.formatting_data(X_query_seqs, X_target_seqs)

    return {
        "inputs": {
            "mirna_int_seq": X_mirna,
            "mrna_int_seq": X_mrna
        },
        "metadata": {
            "mirna_id": np.array(X_query_ids),
            "mrna_id": np.array(X_query_ids)
        }
    }
