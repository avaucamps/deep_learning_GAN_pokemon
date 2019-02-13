import os
from pathlib import Path 
import tensorflow as tf
import numpy as np

class Generator:
    def __init__(self, input_shape, output_shape):        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.training = True
        self.var_scope = 'generator'
    

    def generate(self, inputs, is_training, reuse=False):
        n_output_channels = self.output_shape[2]

        with tf.variable_scope(self.var_scope) as scope:
            if reuse:
                scope.reuse_variables()

            weights_1 = tf.get_variable('weights_1', shape=[self.input_shape, 4 * 4 * 512], 
                        dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias_1 = tf.get_variable('bias_1', shape=[512 * 4 * 4], dtype=tf.float32, 
                        initializer=tf.constant_initializer(0.0))
            flat_conv_1 = tf.add(tf.matmul(inputs, weights_1), bias_1, name='flat_conv_1')
            
            conv_1 = tf.reshape(flat_conv_1, shape=[-1, 4, 4, 512], name='conv_1')
            batch_norm_1 = self.batch_normalization(conv_1, 'batch_norm_1')
            activation_1 = tf.nn.relu(batch_norm_1, "activation_1")

            conv_2 = self.convolution2d_transpose(activation_1, 256, 'conv_2')
            batch_norm_2 = self.batch_normalization(conv_2, 'batch_norm_2')
            activation_2 = tf.nn.relu(batch_norm_2, name='activation_2')

            conv_3 = self.convolution2d_transpose(activation_2, 128, 'conv_3')
            batch_norm_3 = self.batch_normalization(conv_3, 'batch_norm_3')
            activation_3 = tf.nn.relu(batch_norm_3, name='activation_3')

            conv_4 = self.convolution2d_transpose(activation_3, 64, 'conv_4')
            batch_norm_4 = self.batch_normalization(conv_4, 'batch_norm_4')
            activation_4 = tf.nn.relu(batch_norm_4, name='activation_4')

            conv_5 = self.convolution2d_transpose(activation_4, 32, 'conv_5')
            batch_norm_5 = self.batch_normalization(conv_5, 'batch_norm_5')
            activation_5 = tf.nn.relu(batch_norm_5, 'activation_5')

            conv_6 = self.convolution2d_transpose(activation_5, n_output_channels, 'conv_6')
            activation_6 = tf.nn.tanh(conv_6, name='activation_6')

            return activation_6


    def get_trainable_vars(self):
        all_vars = tf.trainable_variables()
        return [var for var in all_vars if self.var_scope in var.name]


    def convolution2d_transpose(self, inputs, filters, name):
        return tf.layers.conv2d_transpose(inputs, filters, kernel_size=[5, 5], strides=[2, 2], 
            padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name=name)


    def batch_normalization(self, inputs, scope):
        return tf.contrib.layers.batch_norm(inputs, is_training=self.training, epsilon=1e-5,
            decay=0.9, updates_collections=None, scope=scope)