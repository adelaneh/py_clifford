#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals

import unittest
from nose.tools import *
import os

from py_clifford.utils.generic_helper import get_install_path

from py_clifford.config import load_configs

datasets_path = os.sep.join([get_install_path(), 'tests', 'test_datasets'])

class ConfigTestCases(unittest.TestCase):
    def setUp(self):
        self.sample_filename = os.sep.join([datasets_path, 'sample_config_1.json'])
        self.sample_config   = load_configs(self.sample_filename)

    def test_load_sample_config(self):
        self.assertEqual(self.sample_config['experiment_name'], 'Sample experiment configuration')
