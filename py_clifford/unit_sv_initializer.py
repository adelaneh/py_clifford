#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.init_ops import Initializer, _assert_float_dtype
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import gen_linalg_ops

@tf_export("initializers.unit_sv_initializer")
class UnitSVInitializer(Initializer):
  """Initializer that generates normally-distributed weights and 
     forces the singular values to one.
  Args:
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  """

  def __init__(self, dtype=dtypes.float32):
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    full_shape = shape if partition_info is None else partition_info.full_shape
    if len(full_shape) != 2:
      raise ValueError(
          "Unit SV matrix initializer can only be used for 2D matrices.")
    if dtype is None:
      dtype = self.dtype
    
    s, u, v = gen_linalg_ops.svd(tf.random.normal(full_shape) / np.sqrt(full_shape[0]))
    eye = linalg_ops_impl.eye(s.shape[0].value, dtype=self.dtype)
    initializer = tf.matmul(u, tf.matmul(eye, v, adjoint_b=True))
    
    if partition_info is not None:
      initializer = array_ops.slice(initializer, partition_info.var_offset,
                                    shape)
    return initializer

  def get_config(self):
    return {"dtype": self.dtype.name}

