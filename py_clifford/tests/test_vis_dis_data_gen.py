#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals

import unittest
from nose.tools import *
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.activations import tanh, linear, relu, sigmoid

from py_clifford.utils.generic_helper import get_install_path
from py_clifford.config import load_configs

from py_clifford.data_generators import *

datasets_path = os.sep.join([get_install_path(), 'tests', 'test_datasets'])

class DataGenerationTestCases(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.config_filename  = os.sep.join([datasets_path, 'test_firing_rate_rnn_config_1.json'])
        self._config          = load_configs(self.config_filename)
        self._data_config     = self._config['data_params']
        self._min_ad          = float(self._data_config['min_angular_diff'])
        self._max_ad          = float(self._data_config['max_angular_diff'])
        self._ann_const       = float(self._data_config['sampling_annealing_const'])
        self._training_config = self._config['training_params']
        self._max_iter        = self._training_config['training_steps']
        self._next_ang_diff   = lambda cur_iter: self._min_ad + (self._max_ad - self._min_ad) * np.exp(-self._ann_const * (cur_iter - 1) / self._max_iter)

    def test_generate_random_orientation_pair(self):
        self.assertEqual(self._next_ang_diff(1),              self._max_ad)
        self.assertEqual(self._next_ang_diff(self._max_iter), self._min_ad)

    def test_generate_trials(self):
        _batch_size        = int(self._training_config['batch_size'])
        _rnd_prob          = float(self._data_config['random_orientation_sampling_prob'])
        _network_config    = self._config['network_params']
        _input_config      = _network_config['input_params']
        _num_inputs        = _input_config['num_orituned_input_units']
        _num_inputs       += 1 if _input_config['has_go_cue_unit'] else 0
        _timesteps         = self._data_config['timesteps']

        _X, _Y, _s1s, _s2s = generate_trials(self._config,
                                             batch_size=_batch_size,
                                             angular_diff_deg=self._next_ang_diff(1),
                                             random_periods=True,
                                             rnd_prob=_rnd_prob,
                                             rescale_input=True,
                                            )
        self.assertEqual(_X.shape, (_batch_size, _timesteps, _num_inputs))

        with self.assertRaises(RuntimeError) as rterr:
            _X, _Y, _s1s, _s2s = generate_trials(self._config,
                                                 batch_size=_batch_size,
                                                 angular_diff_deg=self._next_ang_diff(1),
                                                 random_periods=True,
                                                 rnd_prob=_rnd_prob,
                                                 rescale_input=True,
                                                 angle1_deg=20.,
                                                )

        _X, _Y, _s1s, _s2s = generate_trials(self._config,
                                             batch_size=_batch_size,
                                             angular_diff_deg=self._next_ang_diff(1),
                                             random_periods=True,
                                             rnd_prob=_rnd_prob,
                                             rescale_input=True,
                                             angle1_deg=20.,
                                             angle2_deg=26.,
                                            )
        self.assertEqual(_X.shape, (_batch_size, _timesteps, _num_inputs))
        self.assertAlmostEqual(max(_s1s), float(20./180.*np.pi))
        self.assertAlmostEqual(min(_s1s), float(20./180.*np.pi))
