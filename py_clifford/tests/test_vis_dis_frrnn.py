#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals

import unittest
from nose.tools import *
import os

import tensorflow as tf
import numpy as np

from tensorflow.keras.activations import tanh, linear, relu, sigmoid

from py_clifford.utils.generic_helper import get_install_path
from py_clifford.config import load_configs

from py_clifford.vis_dis_frrnn import VisualDiscriminationFRRNN

datasets_path   = os.sep.join([get_install_path(), 'tests', 'test_datasets'])
models_path     = os.sep.join([get_install_path(), 'tests', 'models'])

class VisualDiscriminationFRRNNTestCases(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self._config_filename = os.sep.join([datasets_path, 'test_firing_rate_rnn_config_2.json'])
        self._config          = load_configs(self._config_filename)
        self._test_save_path  = os.sep.join([models_path, "test_save_load_vis_dis_frrnn_model"])

    def test_create_vis_dis_frrnn(self):
        _visdisfrrnn     = VisualDiscriminationFRRNN(self._config)

        self.assertIsNotNone(_visdisfrrnn)

    def test_train_vis_dis_frrnn(self):
        _visdisfrrnn                    = VisualDiscriminationFRRNN(self._config)
        _visdisfrrnn._training_steps    = 2
        _visdisfrrnn._num_epochs        = 2
        _visdisfrrnn._save_path         = self._test_save_path

        _mean_hidden_bias_before_training   = np.mean(_visdisfrrnn._biases['hidden'].eval(session=_visdisfrrnn._tf_session))
        self.assertEqual(_mean_hidden_bias_before_training, 0.)
#         print('Mean hidden units bias before training = %f'%np.mean(_visdisfrrnn._biases['hidden'].eval(session=_visdisfrrnn._tf_session)))

        _visdisfrrnn.train()

        _mean_hidden_bias_after_training    = np.mean(_visdisfrrnn._biases['hidden'].eval(session=_visdisfrrnn._tf_session))
        self.assertNotEqual(_mean_hidden_bias_after_training, 0.)
#         print('Mean hidden units bias after training = %f'%np.mean(_visdisfrrnn._biases['hidden'].eval(session=_visdisfrrnn._tf_session)))

        self.assertTrue(True)    
    
    def test_test_vis_dis_frrnn(self):
        _visdisfrrnn                    = VisualDiscriminationFRRNN(self._config)
        _visdisfrrnn._training_steps    = 10
        _visdisfrrnn._num_epochs        = 2
        _visdisfrrnn._save_path         = self._test_save_path

        _visdisfrrnn.train()

        _visdisfrrnn._testing_batch_size        = 10
        __output, __hat, __error, __s1s, __s2s, __X, __Y  = _visdisfrrnn.test(45, 60)

        self.assertAlmostEqual(__s1s[0], 45.*np.pi/180)
        self.assertAlmostEqual(__s2s[9], 60.*np.pi/180)

    def test_save_vis_dis_frrnn_model(self):
        _visdisfrrnn                    = VisualDiscriminationFRRNN(self._config)
        _visdisfrrnn._training_steps    = 2
        _visdisfrrnn._num_epochs        = 2
        _visdisfrrnn._save_weights      = False
        _visdisfrrnn._save_path         = self._test_save_path

        _visdisfrrnn.train()

        print("Trying to save the model ...")
        _visdisfrrnn.save_model(self._test_save_path)

        self.assertTrue(True)    

    def test_load_vis_dis_frrnn_model(self):
        _visdisfrrnn                            = VisualDiscriminationFRRNN(self._config)

        _mean_hidden_bias_before_training       = np.mean(_visdisfrrnn._biases['hidden'].eval(session=_visdisfrrnn._tf_session))
        self.assertEqual(_mean_hidden_bias_before_training, 0.)

        print("Trying to load the model from %s ..."%self._test_save_path)
        _visdisfrrnn.load_model(self._test_save_path)

        _mean_hidden_bias_after_training    = np.mean(_visdisfrrnn._biases['hidden'].eval(session=_visdisfrrnn._tf_session))
        self.assertNotEqual(_mean_hidden_bias_after_training, 0.)
 
