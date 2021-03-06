.. image:: https://travis-ci.org/adelaneh/py_clifford.svg?branch=master
  :target: https://travis-ci.org/adelaneh/py_clifford

.. image:: https://ci.appveyor.com/api/projects/status/88ki1hfa9sr7tr3g?svg=true
  :target: https://ci.appveyor.com/project/adelaneh/py-clifford

.. image:: https://coveralls.io/repos/github/adelaneh/py_clifford/badge.svg
  :target: https://coveralls.io/github/adelaneh/py_clifford


py_clifford
=================

This repository provides abstractions and utilities to use firing rate recurrent 
neural networks (FRRNNs) for simulating 2-line visual discrimination task (see 
the reference section below). It also contains data analysis procedures which 
use concepts and techniques from geometrical topology and linear algebra to 
find low-dimensional manifolds in the activity space of the FRRNNs over each trial.

.. image:: https://github.com/adelaneh/py_clifford/blob/master/docs/images/clifford_mov.gif

*More details to come!*

The package is free, open-source, and BSD-licensed.

Important links
===============

* Project Homepage: http://www.columbia.edu/~aa4348/py_clifford
* Code repository: https://github.com/adelaneh/py_clifford
* Issue Tracker: https://github.com/adelaneh/py_clifford/issues

Dependencies
============

The required dependencies to build the packages are:

* pandas (provides data structures to store and manage tables)
* numpy (version 1.16, used to store similarity matrices and required by pandas)
* matplotlib (provides tools to create plots and animations)
* tensorflow (version 1.15, used as the baseline to implement FRRNNs)
* tqdm (facilitates tracking the progress of training and testing the FRRNNs)

Platforms
=========

py_clifford has been tested on Linux, OS X and Windows.

Installation
============

See `installation instructions <docs/user_manual/installation.rst>`_.

Usage
=====

See the accompanying notebooks:

* `Simulating the 2-line Visual Discrimination Experiments using Firing Rate RNNs <notebooks/VisualDiscriminationTaskSimulation.ipynb>`_
* More notebooks to come.

References
==========
For details of the theoretical and experimental background of the psychophysical tasks, see::

    @article {DingE9115,
        author = {Ding, Stephanie and Cueva, Christopher J. and Tsodyks, Misha and Qian, Ning},
        title = {Visual perception as retrospective Bayesian decoding from high- to low-level features},
        volume = {114},
        number = {43},
        pages = {E9115--E9124},
        year = {2017},
        journal = {Proceedings of the National Academy of Sciences}
    }

For details of the simulations and data analysis, watch this page for the soon-to-be-released paper!

