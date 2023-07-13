from kipoi.model import BaseModel

from io import open

from collections import OrderedDict
from collections.abc import Mapping
import sys
import os
import yaml
import kipoi  # for .config module
from kipoi_utils import (load_module, cd, merge_dicts, read_pickle, override_default_kwargs,
                         load_obj, inherits_from, infer_parent_class, makedir_exist_ok)
import abc
import numpy as np
import json
import yaml

import importlib

from kipoi.specs import ModelDescription, RemoteFile, DataLoaderImport, download_default_args
from kipoi.pipeline import Pipeline
import logging
from packaging.version import Version
import tensorflow

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class GradientMixin(object):
    __metaclass__ = abc.ABCMeta
    allowed_functions = ["sum", "max", "min", "absmax"]

    @abc.abstractmethod
    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        """
        Calculate the input-layer gradient for filter `filter_idx` in layer `layer` with respect to `x`. If avg_func
        is defined average over filters with the averaging function `avg_func`. If `filter_idx` and `avg_func` are both
        not None then `filter_idx` is first applied and then `avg_func` across the selected filters.

        # Arguments
            x: model input
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by KerasModel at the moment
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
        """
        raise NotImplementedError


class LayerActivationMixin():
    @abc.abstractmethod
    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        """
        Get predictions based on layer output.

        Arguments:
            x: model inputs from dataloader batch
            layer: layer identifier / name. can be integer or string.
            pre_nonlinearity: Assure that output is returned from before the non-linearity. This feature does
            not have to be implemented always (not possible). If not implemented and set to True either raise
            Error or at least warn!
        """
        raise NotImplementedError


class tfKerasModel(BaseModel, GradientMixin, LayerActivationMixin):
    """Loads the serialized Keras model

    # Arguments
        weights: File path to the hdf5 weights or the hdf5 Keras model
        arch: Architecture json model. If None, `weights` is
    assumed to speficy the whole model
        custom_objects: Python file defining the custom Keras objects
    in a `OBJECTS` dictionary
        backend: Keras backend to use ('tensorflow', 'theano', ...)
        image_dim_ordering: 'tf' or 'th': Whether to use 'tf' ('channels_last')
            or 'th' ('cannels_first') dimension ordering.

    # `model.yml` entry

        ```
        Model:
          type: Keras
          args:
            weights: model.h5
            arch: model.json
            custom_objects: custom_keras_objects.py
        ```
    """

    MODEL_PACKAGE = "tensorflow"

    def __init__(self, weights, arch=None, custom_objects=None, backend=None, image_dim_ordering=None):
        self.backend = backend
        self.image_dim_ordering = image_dim_ordering

        # if keras.backend.backend() == 'tensorflow':
        #     import tensorflow as tf
        #     self.graph = tf.Graph()
        #     self.sess = tf.Session(graph=self.graph)

        #     keras.backend.set_session(self.sess)
        #     keras.backend.clear_session()

        if self.backend is not None and 'KERAS_BACKEND' not in os.environ:
            logger.info("Using Keras backend: {0}".format(self.backend))
            os.environ['KERAS_BACKEND'] = self.backend
        if self.image_dim_ordering is not None:
            import tensorflow.keras.backend as K
            logger.info("Using image_dim_ordering: {0}".format(self.image_dim_ordering))
            try:
                K.set_image_dim_ordering(self.image_dim_ordering)
            except AttributeError:
                if image_dim_ordering != 'tf':
                    raise RuntimeError("only tf dim ordering at is supported")


        from tensorflow import keras
        from tensorflow.keras.models import model_from_json, load_model

        if self.backend is not None:
            if tensorflow.keras.backend.backend() != self.backend:
                logger.warning("Keras backend is {0} instead of {1}".
                               format(tensorflow.keras.backend.backend(), self.backend))

        if custom_objects is not None and os.path.exists(custom_objects):
            self.custom_objects = load_module(custom_objects).OBJECTS
        else:
            self.custom_objects = {}

        self.weights = weights
        self.arch = arch

        self.gradient_functions = {}  # contains dictionaries with string reps of filter functions / slices
        self.activation_functions = {}  # contains the activation functions

        if arch is None:
            # load the whole model
            import tensorflow as tf
            self.model = load_model(weights, custom_objects={'tf':tf})
            logger.info('successfully loaded the model from {}'.
                        format(weights))
        else:
            # load arch
            import tensorflow as tf
            with open(arch, "r", encoding="utf-8") as arch:
                self.model = model_from_json(arch.read(),
                                             custom_objects={'tf':tf})
            logger.info('successfully loaded model architecture from {}'.
                        format(arch))

            # load weights
            self.model.load_weights(weights)
            logger.info('successfully loaded model weights from {}'.
                        format(weights))

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def _does_model_start_with_input_layer(self):
        # the model used to start with an InputLayer,
        # since keras 2.2 (i think, maybe 2.1)
        # this implicit input layer is gone.
        # => to not break anything we might need to compensate for that
        # Consider:
        # "Sequential([Dense(32, input_shape=(784,), activation='relu', name="first")])"
        # for keras 2.0.x we get 2 layers:
        #   -  keras.engine.topology.InputLayer
        #   - keras.layers.core.Dense object
        # for  keras 2.2.x  this will be a single layer
        #   - keras.layers.core.Dense
        import keras
        try:
            first_layer = self.model.get_layer(index=0)
        except:
            # if we cannot get the 0th layer it is for sure no input layer
            return False
        try:
            if isinstance(first_layer, keras.engine.topology.InputLayer):
                return True
            else:
                return False
        except AttributeError:
                # keras does not seem to have this attribute
            return False

    def get_layers_and_outputs(self, layer=None, use_final_layer=False, pre_nonlinearity=False):
        """
        Get layers and outputs either by name / index or from the final layer(s).
        If the final layer should be used it has an activation function that is not Linear, then the input to the
        activation function is returned. This check is not performed when `use_final_layer` is False and `layer`
        is being used.

        Arguments:
            layer: layer index (int) or name (non-int)
            use_final_layer:  instead of using `layer` return the final model layer(s) + outputs
        """
        import keras
        sel_outputs = []
        sel_output_dims = []

        def output_sel(layer, output):
            """
            If pre_nonlinearity is true run get_pre_activation_output
            """
            if pre_nonlinearity:
                output = self.get_pre_activation_output(layer, output)[0]  # always has length 1
            return output

        # If the final layer should be used: (relevant for gradient)
        if use_final_layer:
            # Use outputs from the model output layer(s)
            # If the last layer should be selected automatically then:
            # get all outputs from the model output layers
            if isinstance(self.model, keras.models.Sequential):
                selected_layers = [self.model.layers[-1]]
            else:
                # Model has not attribute output_layers
                try:
                    selected_layers = self.model.output_layers
                except AttributeError:
                    selected_layers = self.model._output_layers
            for l in selected_layers:
                for i in range(self.get_num_inbound_nodes(l)):
                    sel_output_dims.append(len(l.get_output_shape_at(i)))
                    sel_outputs.append(output_sel(l, l.get_output_at(i)))

        # If not the final layer then the get the layer by its name / index
        elif layer is not None:
            if isinstance(layer, int):

                # users / unit tests except layer to be 1 based indexing
                # since the model used to start with an InputLayer
                # since keras 2.2 (i think, maybe 2.1)
                # this implicit input layer is gone.
                # => to not break anything we might need to compensate for that
                input_layer_as_start = self._does_model_start_with_input_layer()

                if input_layer_as_start:
                    selected_layer = self.model.get_layer(index=layer)
                else:
                    selected_layer = self.model.get_layer(index=layer - 1)

            elif isinstance(layer, str):
                selected_layer = self.model.get_layer(name=layer)
            selected_layers = [selected_layer]
            # get the outputs from all nodes of the selected layer (selecting output from individual output nodes
            # creates None entries when running K.gradients())
            if self.get_num_inbound_nodes(selected_layer) > 1:
                logger.warning("Layer %s has multiple input nodes. By default outputs from all nodes "
                               "are concatenated" % selected_layer.name)
                for i in range(self.get_num_inbound_nodes(selected_layer)):
                    sel_output_dims.append(len(selected_layer.get_output_shape_at(i)))
                    sel_outputs.append(output_sel(selected_layer, selected_layer.get_output_at(i)))
            else:
                sel_output_dims.append(len(selected_layer.output_shape))
                sel_outputs.append(output_sel(selected_layer, selected_layer.output))
        else:
            raise Exception("Either use_final_layer has to be set or a layer name has to be defined.")

        return selected_layers, sel_outputs, sel_output_dims

    @staticmethod
    def get_pre_activation_output(layer, output):
        import keras
        # if the current layer uses an activation function then grab the input to the activation function rather
        # than the output from the activation function.
        # This can lead to confusion if the activation function translates to backend operations that are not a
        # single operation. (Which would also be a misuse of the activation function itself.)
        # suggested here: https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
        if hasattr(layer, "activation") and not layer.activation == keras.activations.linear:
            new_output_ois = []
            if hasattr(output, "op"):
                # TF
                for inp_here in output.op.inputs:
                    new_output_ois.append(inp_here)
            else:
                # TH
                # As of the latest version of Theano this feature is not supported - the activation layer is too
                # diffuse to be handeled here since Theano does not have objects for the activation.
                raise Exception("`get_pre_activation_output` is not supported for Theano models!")
                import theano
                for inp_here in output.owner.inputs:
                    if not isinstance(inp_here, theano.gof.Constant):
                        new_output_ois.append(inp_here)
            if len(new_output_ois) > 1:
                raise Exception("More than one input to activation function of selected layer. No general rule "
                                "implemented for handing those cases. Consider using a linear activation function + a "
                                "non-linear activation layer instead.")
            return new_output_ois
        else:
            return [output]

    @staticmethod
    def get_num_inbound_nodes(layer):
        if hasattr(layer, "_inbound_nodes"):
            # Keras 2.1.5
            return len(layer._inbound_nodes)
        elif hasattr(layer, "inbound_nodes"):
            # Keras 2.0.4
            return len(layer.inbound_nodes)
        else:
            raise Exception("No way to find out about number of inbound Nodes")

    @staticmethod
    def homogenize_filter_slices(filter_slices):
        if isinstance(filter_slices, int):
            filter_slices = (filter_slices,)
        if isinstance(filter_slices, slice):
            filter_slices = (filter_slices,)
        if isinstance(filter_slices, list):
            filter_slices = tuple(filter_slices)
        if isinstance(filter_slices, tuple):
            # Add a 0th dimension for samples if obviously missing, but no information about the actual dimensions
            # is known!
            if len(filter_slices) == 1:
                filter_slices = tuple([slice(None)] + list(filter_slices))
        return filter_slices

    def _get_gradient_function(self, layer=None, use_final_layer=False, pre_nonlinearity=False, filter_slices=None,
                               filter_func=None, filter_func_kwargs=None):
        """
        Get keras gradient function

        # Arguments:
            layer: Layer name or index with respect to which the input gradients should be returned
            use_final_layer: Alternative to `layer`, if the final layer should be used. In this case `layer` can be None.
            filter_slices: Selection of filters in `layer` that should be taken into consideration
            filter_func: Function to be applied on all filters of the selected layer. If both `filter_slices` and
                `filter_func` are defined, then `filter_slices` will be applied first and then `filter_func`.
            filter_func_kwargs: keyword argument dict passed on to `filter_func`
        """
        import keras
        import copy
        from keras.models import Model
        # Generate the gradient functions according to the layer / filter definition
        gradient_function = None

        layer_label = layer
        # Try to use a previously generated gradient function
        if use_final_layer:
            layer_label = "_KIPOI_FINAL_"

        if layer_label is None:
            raise Exception("Either `layer` must be defined or `use_final_layer` set to True.")

        # Cannot query the layer output shape, so only if the slice is an integer or a list of length 1 it is
        # clear that the batch dimension is missing
        if filter_slices is not None:
            filter_slices = self.homogenize_filter_slices(filter_slices)

        if layer_label not in self.gradient_functions:
            self.gradient_functions[layer_label] = {}
        filter_id = str(filter_slices) + "_PNL_" + str(pre_nonlinearity)
        if filter_func is not None:
            filter_id = str(filter_func) + ":" + str(filter_func_kwargs) + ":" + filter_id
        if filter_id in self.gradient_functions[layer_label]:
            gradient_function = self.gradient_functions[layer_label][filter_id]

        if gradient_function is None:
            # model layer outputs wrt which the gradient shall be calculated
            selected_layers, sel_outputs, sel_output_dims = self.get_layers_and_outputs(layer=layer,
                                                                                        use_final_layer=use_final_layer,
                                                                                        pre_nonlinearity=pre_nonlinearity)

            # copy the model input in case learning flag has to appended when using the gradient function.
            inp = copy.copy(self.model.inputs)

            # Now check if layer outputs have to be concatenated (multiple input nodes in the respective layer)
            has_concat_output = False
            if len(sel_outputs) > 1:
                has_concat_output = True
                # Flatten layers in case dimensions don't match
                all_filters_flat = [keras.layers.Flatten()(x) if dim > 2 else x for x, dim in
                                    zip(sel_outputs, sel_output_dims)]
                # A new model has to be generated in order for the concatenated layer output to have a defined layer output
                if hasattr(keras.layers, "Concatenate"):
                    # Keras 2
                    all_filters_merged = keras.layers.Concatenate(axis=-1)(all_filters_flat)
                    gradient_model = Model(inputs=inp, outputs=all_filters_merged)
                else:
                    # Keras 1
                    all_filters_merged = keras.layers.merge(all_filters_flat, mode='concat')
                    gradient_model = Model(input=inp, output=all_filters_merged)
                # TODO: find a different way to get layer outputs...
                # gradient_model.compile(optimizer=self.model.optimizer, loss=self.model.loss)
                gradient_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
                # output of interest for a given gradient
                output_oi = gradient_model.output
            else:
                output_oi = sel_outputs[0]

            # Which subset of the selected layer outputs should be looked at?
            if filter_slices is not None:
                if has_concat_output:
                    logger.warning("Filter slices have been defined for output selection from layers %s, but "
                                   "layer outputs of nodes had to be concatenated. This will potentially lead to undesired "
                                   "output - please take this concatenation into consideration when "
                                   "defining `filter_slices`." % str([l.name for l in selected_layers]))
                output_oi = output_oi[filter_slices]

            # Should a filter function be applied
            if filter_func is not None:
                if filter_func_kwargs is None:
                    filter_func_kwargs = {}
                output_oi = filter_func(output_oi, **filter_func_kwargs)

            if (filter_slices is None) and (filter_func is None):
                raise Exception("Either filter_slices or filter_func have to be set!")

            # generate the actual gradient function
            from keras import backend as K
            saliency = K.gradients(output_oi, inp)

            if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inp.append(K.learning_phase())

            gradient_function = K.function(inp, saliency)

            # store the generated gradient function:
            self.gradient_functions[layer_label][filter_id] = gradient_function

        return gradient_function

    def _get_feed_input_names(self):
        import keras
        from keras import backend as K
        feed_input_names = None
        if keras.__version__[0] == '1':
            feed_input_names = self.model.input_names
        else:
            if not hasattr(self.model, "_feed_input_names"):
                if not self.model.built:
                    self.model.build()
            feed_input_names = self.model._feed_input_names
        return feed_input_names

    @staticmethod
    def _get_standardize_input_data_func():
        import keras
        if keras.__version__[0] == '1':
            from keras.engine.training import standardize_input_data as _standardize_input_data
        elif hasattr(keras.engine.training, "_standardize_input_data"):
            from keras.engine.training import _standardize_input_data
        elif hasattr(keras.engine.training_utils, "standardize_input_data"):
            from keras.engine.training_utils import standardize_input_data as _standardize_input_data
        else:
            raise Exception("This Keras version is not supported!")
        return _standardize_input_data

    def _batch_to_list(self, x):
        import keras
        from keras import backend as K
        feed_input_names = self._get_feed_input_names()

        # depending on the version this function needs to be imported from different places
        _standardize_input_data = KerasModel._get_standardize_input_data_func()

        if keras.__version__[0] == '1':
            from keras.engine.training import standardize_input_data as _standardize_input_data
            if not self.model.built:
                self.model.build()
            iis = None
            if hasattr(self.model, "internal_input_shapes"):
                iis = self.model.internal_input_shapes
            elif hasattr(self.model, "model") and hasattr(self.model.model, "internal_input_shapes"):
                iis = self.model.model.internal_input_shapes
            x_standardized = _standardize_input_data(x, feed_input_names,
                                                     iis)
        else:
            if not hasattr(self.model, "_feed_input_names"):
                if not self.model.built:
                    self.model.build()
            fis = None
            if hasattr(self.model, "_feed_input_shapes"):
                fis = self.model._feed_input_shapes
            x_standardized = _standardize_input_data(x, feed_input_names, fis)

        return x_standardized

    def _match_to_input(self, to_match, input):
        feed_input_names = self._get_feed_input_names()
        if isinstance(input, np.ndarray):
            assert len(to_match) == 1
            outputs = to_match[0]
        elif isinstance(input, list):
            # Already in right format
            outputs = to_match
        elif isinstance(input, dict):
            from collections import OrderedDict
            outputs_dict = OrderedDict()
            for k, v in zip(feed_input_names, to_match):
                outputs_dict[k] = v
            outputs = outputs_dict
        return outputs

    def _input_grad(self, x, layer=None, use_final_layer=False, filter_slices=None,
                    filter_func=None, filter_func_kwargs=None, pre_nonlinearity=False):
        """Adapted from keras.engine.training.predict_on_batch. Returns gradients for a single batch of samples.

        # Arguments
            x: Input samples, as a Numpy array.

        # Returns
            Numpy array(s) of predictions.
        """
        import keras
        from keras import backend as K
        x_standardized = self._batch_to_list(x)
        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x_standardized + [0.]
        else:
            ins = x_standardized
        gf = self._get_gradient_function(layer, use_final_layer=use_final_layer, filter_slices=filter_slices,
                                         filter_func=filter_func, filter_func_kwargs=filter_func_kwargs,
                                         pre_nonlinearity=pre_nonlinearity)
        outputs = gf(ins)

        # re-format to how the input was:
        return self._match_to_input(outputs, x)

    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        """
        Calculate the input-layer gradient for filter `filter_idx` in layer `layer` with respect to `x`. If avg_func
        is defined average over filters with the averaging function `avg_func`. If `filter_idx` and `avg_func` are both
        not None then `filter_idx` is first applied and then `avg_func` across the selected filters.

        # Arguments
            x: model input
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by KerasModel at the moment
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
        """
        import tensorflow.keras.tensorflow.backend as K
        _avg_funcs = {"sum": K.sum, "min": K.min, "max": K.max, "absmax": lambda x: K.max(K.abs(x))}
        if avg_func is not None:
            assert avg_func in _avg_funcs
            avg_func = _avg_funcs[avg_func]
        else:
            try:
                backend = K._BACKEND
            except Exception:
                backend = K.backend()
            if backend == "theano":
                avg_func = _avg_funcs["sum"]
        if selected_fwd_node is not None:
            raise Exception("'selected_fwd_node' is currently not supported for Keras models!")
        return self._input_grad(x, layer=layer, filter_slices=filter_idx, use_final_layer=final_layer,
                                filter_func=avg_func, pre_nonlinearity=pre_nonlinearity)

    def _generate_activation_output_functions(self, layer, pre_nonlinearity):
        import copy
        layer_id = str(layer) + "_" + str(pre_nonlinearity)
        if layer_id in self.activation_functions:
            return self.activation_functions[layer_id]

        # get the selected layers
        selected_layers, sel_outputs, sel_output_dims = self.get_layers_and_outputs(layer=layer,
                                                                                    use_final_layer=False,
                                                                                    pre_nonlinearity=pre_nonlinearity)

        # copy the model input in case learning flag has to appended when using the activation function.
        inp = copy.copy(self.model.inputs)

        # Can't we have multiple outputs for the function?
        output_oi = sel_outputs  # list of outputs should work: https://keras.io/backend/#backend-functions -> backend.function

        from keras import backend as K

        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inp.append(K.learning_phase())

        activation_function = K.function(inp, output_oi)

        # store the generated activation function:
        self.activation_functions[layer_id] = activation_function

        return activation_function

    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        """Adapted from keras.engine.training.predict_on_batch. Returns gradients for a single batch of samples.

        Arguments
            x: Input samples, as a Numpy array.

        Returns
            Numpy array(s) of predictions.
        """
        import keras
        from keras import backend as K

        # depending on the keras version this functions needs to be imported from
        # different places
        _standardize_input_data = KerasModel._get_standardize_input_data_func()
        if keras.__version__[0] == '1':
            x = _standardize_input_data(x, self.model.input_names,
                                        self.model.internal_input_shapes)
        else:
            x = _standardize_input_data(x, self.model._feed_input_names,
                                        self.model._feed_input_shapes)

        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + [0.]
        else:
            ins = x
        af = self._generate_activation_output_functions(layer, pre_nonlinearity)
        outputs = af(ins)
        return outputs
