import numpy as np
# import pandas as pd
from kipoi.model import BaseModel
from mmsplice import MMSplice
from mmsplice.utils import predict_deltaLogitPsi
# , read_ref_psi_annotation, \
#     delta_logit_PSI_to_delta_PSI
from mmsplice.mtsplice import MTSplice  # , tissue_names


mmsplice = MMSplice()
mtsplice = MTSplice()


class MMSpliceModel(BaseModel):
    '''Model to predict delta logit PSI'''

    def predict_on_batch(self, inputs):
        X_ref = mmsplice.predict_on_batch(inputs['seq'])
        X_alt = mmsplice.predict_on_batch(inputs['mut_seq'])
        delta_logit_psi = predict_deltaLogitPsi(X_ref, X_alt)

        X_tissue = mtsplice.predict_on_batch(inputs['tissue_seq'])
        X_tissue += np.expand_dims(delta_logit_psi, axis=1)

        # if natural_scale:
        #     df = pd.concat([df, tissue_pred], axis=1)
        #     tissue_pred = pd.DataFrame(X_tissue, columns=tissue_names)
        #     ref_psi_version = 'grch37'
        #     natural_scale = True

        #     df_ref = read_ref_psi_annotation(
        #         ref_psi_version, {'chr1'})

        #     df_ref = df_ref[df_ref.columns[6:]]
        #     df = df.join(df_ref, on='exons', rsuffix='_ref')

        #     delta_logit_PSI_to_delta_PSI(
        #         df[df_ref.columns].values,
        #         df[ref_tissue_names].values
        #     )
        #     df.reset_index(drop=True, inplace=True)
        #     delta_psi_pred.reset_index(drop=True, inplace=True)
        #     df = pd.concat([df, delta_psi_pred], axis=1)

        return X_tissue
