#!/usr/bin/env python
# coding: utf-8

import json
import os

"""Loading configs from JSON file.

    Args:
        filename: The name of the JSON config file.

    Returns:
        A dictionary contraining the configs read from filename.

    Raises:
        AssertionError: If filename is empty.
        AssertionError: If file named filename does not exist.
"""
def load_configs(filename):
    if filename is None or filename == '':
        raise AssertionError('Invalid config file name.')
    if not os.path.isfile(filename):
        raise AssertionError('File does not exist.')

    with open(filename) as ff:
        config = json.load(ff)

    return config
