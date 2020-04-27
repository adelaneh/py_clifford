#!/usr/bin/env python
# coding: utf-8

from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

import numpy as np

_WEIGHTS_VARIABLE_NAME = "hidden_weights"
_BIAS_VARIABLE_NAME = "hidden_biases"

_DT  = 1.
_TAU = 10.
_DT_OVER_TAU = _DT / _TAU

@tf_export("nn.rnn_cell.FiringRateRNNCell")
class FiringRateRNNCell(LayerRNNCell):
    """Firing rate RNN cell. the dynamics are determined by the following difference equation:

    .. math:: s^{(t+1)} = \\frac{d{t}}{\\tau} \Big(W \\big(f(s^{(t)}) + \gamma^{(t)}\\big) + A x^{(t)} + b\Big) + \Big(1-\\frac{d{t}}{\\tau}\Big) s^{(t)}

    See the arXiv paper for more details.
    
    Args:
        num_units: int, The number of units in the RNN cell.
        hidden_units_noise_std: float, magnitude of noise injected into hidden units
        droupout:
        activation: nonlinearity to use.  Default: `tanh`.
        reuse: (optional) python boolean describing whether to reuse variables in an 
            existing scope.  If not `True`, and the existing scope already has the 
            given variables, an error is raised.
        name: string, the name of the layer. Layers with the same name will share 
            weights, but to avoid mistakes we require reuse=True in such cases.
        dtype: default dtype of the layer (default of `None` means use the type of 
            the first input). Required when `build` is called before `call`.
    """

    def __init__(self,
               num_units,
               activation=math_ops.tanh,
               dtovertau=_DT_OVER_TAU,
               w_initializer=None,
               hidden_units_noise_std=0.,
               layer_normalize=False,
               reuse=None,
               name=None,
               dtype=None
              ):
        super(FiringRateRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

        if num_units is None:
            raise TypeError("No number of units provided.")
        if num_units <= 0:
            raise ValueError("Invalid number of units (%d)."%num_units)
        if activation is None:
            raise TypeError("No activation function provided.")
        if dtovertau is None:
            raise TypeError("No dt-over-tau provided.")
        if dtovertau <= 0:
            raise ValueError("Invalid dt-over-tau (%f)."%dtovertau)
        if hidden_units_noise_std is None:
            raise TypeError("No hidden unit noise std provided.")
        if hidden_units_noise_std < 0:
            raise ValueError("Invalid hidden unit noise std (%f)."%hidden_units_noise_std)
        if layer_normalize is None:
            raise TypeError("No layer normalization indicator provided.")

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units                 = num_units
        self._activation                = activation
        self._dtovertau                 = dtovertau
        self._w_initializer             = w_initializer
        self._hidden_units_noise_std    = hidden_units_noise_std
        self._layer_normalize           = layer_normalize

        self._is_generate_noise         = True

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
    
        self._input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[self._input_depth + self._num_units, self._num_units],
            initializer=self._w_initializer(dtype=self.dtype) if self._w_initializer is not None else None,
        )
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype),
        )

        self.built = True

    def call(self, inputs, state): # TODO: Make noise an argument to "call"
        """Firing rate model RNN: 
        
        `new_gate_inputs = dt_over_tau*(W * input + U * state + B)+(1 - dt_over_tau)*gate_inputs`

        `output = new_state = act(new_gate_inputs)`

        ..    Each row of the inputs consists of (1) a network input of dimension self._input_depth, and (2) hidden state noise values. These 
        """
        ## Add noise to the current state
        if self._hidden_units_noise_std > 0:
            # noise              = random_ops.random_normal(array_ops.shape(state),mean = 0.0,  stddev=self._hidden_units_noise_std)
            if self._is_generate_noise:
                noise                   = np.random.normal(0.0, self._hidden_units_noise_std, state.shape)
                self._is_generate_noise = False
            postactnoise_state = math_ops.add(self._activation(state), noise)
        else:
            postactnoise_state = self._activation(state)
   
        #one timestep calculation
        if self._layer_normalize:
            prenorm_state = math_ops.matmul(
                                array_ops.concat([inputs, postactnoise_state], 1),
                                self._kernel
                            )
            prenorm_state_mu, prenorm_state_var = tf.nn.moments(prenorm_state, [1], keepdims=True)
            postnorm_state = (prenorm_state - prenorm_state_mu) / tf.sqrt(prenorm_state_var + 1e-3)

            state = self._dtovertau * nn_ops.bias_add(postnorm_state, self._bias) + \
                    (1. - self._dtovertau) * state
        else:
            state = self._dtovertau * nn_ops.bias_add(
                                                math_ops.matmul(
                                                    array_ops.concat([inputs, postactnoise_state], 1), 
                                                    self._kernel
                                                ), 
                                                self._bias) + \
                    (1. - self._dtovertau) * state

        return state, state
