# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_impl


class Dense(core_layers.Dense):
    def __init__(self, *args, **kwargs):
        self.weight_norm = kwargs.pop("weight_norm")
        super(Dense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})
        kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.weight_norm:
            self.g = self.add_variable(
                "wn/g",
                shape=(self.units,),
                initializer=init_ops.ones_initializer(),
                dtype=kernel.dtype,
                trainable=True)
            self.kernel = nn_impl.l2_normalize(kernel, axis=0) * self.g
        else:
            self.kernel = kernel
        if self.use_bias:
            self.bias = self.add_variable(
                'bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True


def dense(
        inputs, units,
        activation=None,
        weight_norm=True,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    layer = Dense(units,
                  activation=activation,
                  weight_norm=weight_norm,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer,
                  activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint,
                  bias_constraint=bias_constraint,
                  trainable=trainable,
                  name=name,
                  dtype=inputs.dtype.base_dtype,
                  _scope=name,
                  _reuse=reuse)
    return layer.apply(inputs)
