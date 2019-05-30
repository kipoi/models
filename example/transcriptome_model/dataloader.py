from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd


def load_gene_expression(csv_file, expected_columns=None):
    """Load gene expression values

    Args:
      csv_file: containing gene expression values. First column contains sample names
      expected_columns: txt file containing gene names
    """
    df = pd.read_csv(csv_file, index_col=0)

    columns = np.loadtxt(expected_columns, dtype=str)

    if list(df.columns) != list(columns):
        raise ValueError("Column names don't match the expected"
                         " column names stored in file: {}".format(expected_columns))

    return {
        "inputs": df.values,
        "metadata": {
            "sample_id": df.index.values
        }
    }
