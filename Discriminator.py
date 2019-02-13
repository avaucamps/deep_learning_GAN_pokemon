import os
from pathlib import Path 
import tensorflow as tf
from utils import lrelu
import numpy as np


class Discriminator:
    def __init__(self):
        self.training = True
        self.var_scope = 'discriminator'


    def discriminate(self, inputs, is_training, reuse=False):
        with tf.variable_scope(self.var_scope) as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = self.conv(inputs, 64, 'conv1')
            batch_norm_1 = self.batch_norm(conv1, 'batch_norm1')
            activation1 = lrelu(batch_norm_1, n='activation1')
            
            conv2 = self.conv(activation1, 128, 'conv2')
            batch_norm_2 = self.batch_norm(conv2, 'batch_norm2')
            activation2 = lrelu(batch_norm_2, n='activation2')

            conv3 = self.conv(activation2, 256, 'conv3')
            batch_norm_3 = self.batch_norm(conv3, 'batch_norm3')
            activation3 = lrelu(batch_norm_3, n='activation3')

            conv4 = self.conv(activation3, 512, 'conv4')
            batch_norm_4 = self.batch_norm(conv4, 'batch_norm4')
            activation4 = lrelu(batch_norm_4, n='activation4')

            feature_dimension = np.prod(activation4.get_shape()[1:])
            fully_connected_1 = tf.reshape(activation4, shape=[-1, feature_dimension], name='fully_connected_1')

            weights2 = tf.get_variable('weights2', shape=[fully_connected_1.shape[-1], 1], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias2 = tf.get_variable('bias2', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            logits = tf.add(tf.matmul(fully_connected_1, weights2), bias2, name='logits')

            return logits


    def get_trainable_vars(self):
        all_vars = tf.trainable_variables()
        return [var for var in all_vars if self.var_scope in var.name]


    def conv(self, inputs, filters, name):
        return tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[5,5],
                strides=[2,2],
                padding="SAME",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name=name
        )


    def batch_norm(self, inputs, scope):
        return tf.contrib.layers.batch_norm(
            inputs, 
            is_training=self.training, 
            epsilon=1e-5, 
            decay = 0.9,  
            updates_collections=None,
            scope=scope
        )