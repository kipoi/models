import numpy as np
import tensorflow as tf

class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

class StochasticReverseComplement(tf.keras.layers.Layer):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, **kwargs):
        super(StochasticReverseComplement, self).__init__()

    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)


class SwitchReverse(tf.keras.layers.Layer):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self, **kwargs):
        super(SwitchReverse, self).__init__()

    def call(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.shape)
        if xd == 3:
            rev_axes = [1]
        elif xd == 4:
            rev_axes = [1, 2]
        else:
            raise ValueError("Cannot recognize SwitchReverse input dimensions %d." % xd)

        return tf.keras.backend.switch(reverse, tf.reverse(x, axis=rev_axes), x)


class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad="uniform", **kwargs):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.augment_shifts = tf.range(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(
                shape=[], minval=0, dtype=tf.int64, maxval=len(self.augment_shifts)
            )
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(
                tf.not_equal(shift, 0),
                lambda: shift_sequence(seq_1hot, shift),
                lambda: seq_1hot,
            )
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shift_max": self.shift_max, "pad": self.pad})
        return config


def shift_sequence(seq, shift, pad_value=0.25):
    """Shift a sequence left or right by shift_amount.

    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0 : tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(
        tf.greater(shift, 0), lambda: _shift_right(seq), lambda: _shift_left(seq)
    )
    sseq.set_shape(input_shape)

    return sseq


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    activation="gelu",
    activation_end=None,
    strides=1,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    conv_type="standard",
    residual=False,
    pool_size=1,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
    padding="same",
):
    """Construct a single convolution block.
    Args:
        inputs:        [batch_size, seq_length, features] input sequence
        filters:       Conv1D filters
        kernel_size:   Conv1D kernel_size
        activation:    relu/gelu/etc
        strides:       Conv1D strides
        dilation_rate: Conv1D dilation rate
        l2_scale:      L2 regularization weight.
        dropout:       Dropout rate probability
        conv_type:     Conv1D layer type
        residual:      Residual connection boolean
        pool_size:     Max pool width
        batch_norm:    Apply batch normalization
        bn_momentum:   BatchNorm momentum
        bn_gamma:      BatchNorm gamma (defaults according to residual)
      Returns:
        [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    if activation=="gelu":
        current = GELU()(current)
    else:
        current = tf.keras.layers.ReLU()(current)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # Pool
    if pool_size > 1:
        current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
            current
        )

    return current


def conv_tower(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    divisible_by=1,
    repeat=1,
    **kwargs
):
    """Construct a reducing convolution block.
    Args:
        inputs:        [batch_size, seq_length, features] input sequence
        filters_init:  Initial Conv1D filters
        filters_end:   End Conv1D filters
        filters_mult:  Multiplier for Conv1D filters
        divisible_by:  Round filters to be divisible by (eg a power of two)
        repeat:        Tower repetitions
    Returns:
        [batch_size, seq_length, features] output sequence
    """

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    # determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        # convolution
        current = conv_block(current, filters=_round(rep_filters), **kwargs)

        # update filters
        rep_filters *= filters_mult

    return current


def dense_block(
    inputs,
    units=None,
    activation="gelu",
    activation_end=None,
    flatten=False,
    dropout=0,
    l2_scale=0,
    l1_scale=0,
    residual=False,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
):
    """Construct a single convolution block.
    Args:
        inputs:         [batch_size, seq_length, features] input sequence
        units:          Conv1D filters
        activation:     relu/gelu/etc
        activation_end: Compute activation after the other operations
        flatten:        Flatten across positional axis
        dropout:        Dropout rate probability
        l2_scale:       L2 regularization weight.
        l1_scale:       L1 regularization weight.
        residual:       Residual connection boolean
        batch_norm:     Apply batch normalization
        bn_momentum:    BatchNorm momentum
        bn_gamma:       BatchNorm gamma (defaults according to residual)
    Returns:
        [batch_size, seq_length(?), features] output sequence
    """
    current = inputs

    if units is None:
        units = inputs.shape[-1]

    # activation
    if activation=="gelu":
        current = GELU()(current)
    else:
        current = tf.keras.layers.ReLU()(current)

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=(not batch_norm),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    return current


def final(
    inputs,
    units,
    activation="linear",
    flatten=False,
    kernel_initializer="he_normal",
    l2_scale=0,
    l1_scale=0,
):
    """Final simple transformation before comparison to targets.
    Args:
        inputs:         [batch_size, seq_length, features] input sequence
        units:          Dense units
        activation:     relu/gelu/etc
        flatten:        Flatten positional axis.
        l2_scale:       L2 regularization weight.
        l1_scale:       L1 regularization weight.
    Returns:
        [batch_size, seq_length(?), units] output sequence
    """
    current = inputs

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)
    return current


OBJECTS = {"GELU": GELU,
           'StochasticReverseComplement': StochasticReverseComplement,
           'SwitchReverse': SwitchReverse,
           'StochasticShift': StochasticShift,
           'shift_sequence': shift_sequence,
           'conv_block': conv_block,
           'conv_tower': conv_tower,
           'dense_block': dense_block,
           'final': final
          }
