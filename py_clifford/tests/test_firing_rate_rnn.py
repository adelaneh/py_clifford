#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals

import unittest
from nose.tools import *
import os

import tensorflow as tf
from tensorflow.keras.activations import tanh, linear, relu, sigmoid

from py_clifford.utils.generic_helper import get_install_path
from py_clifford.config import load_configs

from py_clifford.firing_rate_rnn import FiringRateRNNCell

datasets_path = os.sep.join([get_install_path(), 'tests', 'test_datasets'])

class FiringRateRNNTestCases(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.config_filename = os.sep.join([datasets_path, 'test_firing_rate_rnn_config_1.json'])
        self._config      = load_configs(self.config_filename)
        self.frrnn_config = self._config['network_params']

    def test_create_firing_rate_rnn_cell(self):
        _hidden_layer_config = self.frrnn_config['hidden_params']
        _frrnn               = FiringRateRNNCell(
                                    _hidden_layer_config['num_hidden_units'],
                                    activation=_hidden_layer_config['activation_function'],
                                    dtovertau=_hidden_layer_config['dtovertau'],
                                    w_initializer=None,
                                    layer_normalize=_hidden_layer_config['layer_normalize'],
#                                     reuse=True,
                               )
        self.assertIsNotNone(_frrnn)

    def test_setup_firing_rate_rnn(self):
        _input_config        = self.frrnn_config['input_params']
        _num_inputs          = _input_config['num_orituned_input_units']
        _num_inputs         += 1 if _input_config['has_go_cue_unit'] else 0

        _hidden_layer_config    = self.frrnn_config['hidden_params']
        _num_hidden_units       = _hidden_layer_config['num_hidden_units']
        _hidden_activation_func = eval(_hidden_layer_config['activation_function'])

        _output_config       = self.frrnn_config['output_params']
        _num_outputs         = _output_config['num_sincos_output_units'] + _output_config['num_ordinal_output_units']

        _data_config         = self._config['data_params']

        X = tf.placeholder("float", [None,
                                     _data_config['timesteps'], 
                                     _num_inputs],
                                     name = "X")
        Y = tf.placeholder("float", [None, 
                                     _data_config['timesteps'], 
                                     _num_outputs], 
                                     name = "Y")

        _frrnn           = FiringRateRNNCell(
                                    _hidden_layer_config['num_hidden_units'],
                                    activation=_hidden_activation_func,
                                    dtovertau=float(_hidden_layer_config['dtovertau']),
                                    w_initializer=None,
                                    layer_normalize=bool(_hidden_layer_config['layer_normalize']),
#                                     reuse=True,
                                   )
        _hidden_outputs, _hidden_states = tf.nn.dynamic_rnn(_frrnn, X, dtype=tf.float32)

        # Define weights
        _weights = {
            'hidden': _frrnn._kernel,
            'out': tf.Variable(tf.random_normal([_num_hidden_units, _num_outputs]))
        }
        biases = {
            'hidden': _frrnn._bias,
            'out': tf.Variable(tf.random_normal([_num_outputs]))
        }

        _hidden_after_act = tf.reshape(_hidden_activation_func(
                                            _hidden_outputs,
                                           ),
                                           [-1, _num_hidden_units]
                                      )
        _hidden_before_act = tf.reshape(_hidden_outputs,[-1, _num_hidden_units])

        self.assertTrue(True)
