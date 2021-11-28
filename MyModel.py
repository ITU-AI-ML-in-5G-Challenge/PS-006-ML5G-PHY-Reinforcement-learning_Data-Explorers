import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
    
    @tf.function
    def predict_test(self, inputs):
        t = self.model(np.atleast_2d(inputs.astype('float32')))
        t_reshaped = tf.reshape(t, [3, 64])
        return np.unravel_index(np.argmax(t_reshaped, axis=None), t_reshaped.shape), 0 #return the index of max Q value