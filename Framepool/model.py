from kipoi.model import BaseModel
from keras.models import load_model
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np


class FrameSliceLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        shape = K.shape(x)
        x = K.reverse(x, axes=1)  # reverse, so that frameness is related to fixed point
        frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
        frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
        frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)
        return [frame_1, frame_2, frame_3]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], None, input_shape[2]), (input_shape[0], None, input_shape[2]),
                (input_shape[0], None, input_shape[2])]


class UTRVariantEffectModel(BaseModel):

    def __init__(self, weights):
        self.nuc_dict = {'A': [1.0, 0.0, 0.0, 0.0], 'C': [0.0, 1.0, 0.0, 0.0], 'G': [0.0, 0.0, 1.0, 0.0],
                         'U': [0.0, 0.0, 0.0, 1.0], 'T': [0.0, 0.0, 0.0, 1.0],
                         'N': [0.0, 0.0, 0.0, 0.0], 'X': [1 / 4, 1 / 4, 1 / 4, 1 / 4]}
        self.weights = weights
        self.model = load_model(weights, custom_objects={'FrameSliceLayer': FrameSliceLayer})

    # One-hot encodes a particular sequence
    def encode_seq(self, seq, max_len):
        # Add padding:
        if max_len == 0:
            padding_needed = 0
        else:
            padding_needed = max_len - len(seq)
        seq = seq.upper()  # force upper case to be sure!
        seq = "N" * padding_needed + seq
        # One hot encode
        try:
            one_hot = np.array([self.nuc_dict[x] for x in seq])  # get stacked on top of each other
        except KeyError as e:
            raise ValueError('Cant one-hot encode unkown base: {} in seq: {}. \
                                 Possible cause: a variant in the vcf file is defined by tag (<..>). \
                                 If so, please filter'.format(str(e), seq))
        return one_hot

    # One-hot encodes the entire tensor
    def encode(self, inputs):
        # One Hot Encode input
        max_len = len(max(inputs, key=len))
        one_hot = np.stack([self.encode_seq(seq, max_len)
                            for seq in inputs], axis=0)
        return one_hot

    # Predicts for a batch of inputs
    def predict_on_batch(self, inputs):
        ref_seq = inputs["ref_seq"]
        alt_seq = inputs["alt_seq"]

        if np.isscalar(ref_seq) :
            ref_seq = [ref_seq]
            alt_seq = [alt_seq]

        assert len(ref_seq) == len(alt_seq)

        # Encode
        # max_len = len(max(inputs, key=len))
        one_hot_ref = self.encode(ref_seq)
        one_hot_alt = self.encode(alt_seq)

        # Construct dummy library indicator
        indicator = np.zeros((len(ref_seq), 2))
        indicator[:, 1] = 1

        # Compute fold change for all three frames
        fc_changes = []
        for shift in range(3):
            if shift > 0:
                shifter = np.zeros((one_hot_ref.shape[0], 1, 4))
                one_hot_ref = np.concatenate([one_hot_ref, shifter], axis=1)
                one_hot_alt = np.concatenate([one_hot_alt, shifter], axis=1)
            pred_ref = self.model.predict_on_batch([one_hot_ref, indicator]).reshape(-1)
            pred_variant = self.model.predict_on_batch([one_hot_alt, indicator]).reshape(-1)
            fc_changes.append(np.log2(pred_variant / pred_ref))
        # Return
        return {
            "mrl_fold_change": fc_changes[0],
            "shift_1": fc_changes[1],
            "shift_2": fc_changes[2]
        }
