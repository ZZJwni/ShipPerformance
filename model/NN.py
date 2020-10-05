from pathlib import Path
from typing import List, Dict, Tuple
from collections import namedtuple

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

layer_param = namedtuple(
    'layer_param', ('units', 'activation', 'dropout_rate'))

class NN(object):
    """Class of Dense Neural Network
    """

    def __init__(self, input_size:int, output_size:int, layer_params:List[layer_param],
                 loss=tf.losses.mean_squared_error, optimizer=tf.optimizers.SGD(learning_rate=.001),
                 metrics=None):

        self.input_size = input_size
        self.output_size = output_size
        self.layer_params = layer_params

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self._model = None
        self._history = None
    
    def build(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten())
        for layer_param in self.layer_params:
            model.add(keras.layers.Dense(units=layer_param.units, activation=layer_param.activation))
            if layer_param.dropout_rate is not None:
                model.add(keras.layers.Dropout(rate=layer_param.dropout_rate))
        model.add(keras.layers.Dense(self.output_size))
        model.build(input_shape=(None, self.input_size))
        self._model = model
    
    def train(self,
              X_train, y_train,
              X_val, y_val,
              callbacks=None, batch_size=32, epochs=1000, shuffle=True):
        self._model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self._model.fit(x=X_train, y=y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(X_val, y_val),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        self._history = self._model.history.history
    
    def predict(self, X_test, rescaler=None):

        y_test_pred = self._model.predict(X_test)
        if rescaler is not None:
            y_test_pred = rescaler.inverse_transform(np.concatenate((X_test, y_test_pred), axis=1))[:, -1]
        return y_test_pred.flatten()
    
    def delete(self):
        keras.backend.clear_session()
        self._model = None
        self._history = None
    
    
    