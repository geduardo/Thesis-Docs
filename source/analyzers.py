import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import layers
from keras.models import load_model
class Two_Layers_single_output(object):
    def __init__(self, input_size, learning_rate=0.001):
        """ This class is just a two-layer Keras neural network to process the
        data taken by the experimenter to produce a single output.

        :param input_size: Number of inputs feeded to the analyzer
        :type input_size: int
        :param learning_rate: Gradient descent (Adam) learning rate, defaults 
                              to 0.001
        :type learning_rate: float, optional
        """        
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """Uses keras to build de model. You should modify here the code to
        change the neural network architecture. Default to a 
        *input_sizex16x16x1* fully connected sequential network.
        
        :return: Keras model for the agent using the specified structure. 
        :rtype: keras model
        """        
        model = Sequential([
        layers.Dense(16,activation = 'relu', input_shape=[self.input_size]),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(1)
        ])
        optimizer = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model
    def train(self, X_train, y_train):
        """ Fit the model with the given data an target.
        
        :param X_train: Input data for the model
        :type X_train: Keras valid input data 
        :param y_train: Target value for the input data
        :type y_train: numpy array / float
        """        
        self.model.fit(X_train, y_train, batch_size=1, epochs = 1, verbose = 0)
        
    def predict(self, X_test):
        """ Uses the model to predict a value
        
        :param X_test: Input data for the model
        :type X_test: Keras valid input data
        :return: Prediction of the target value
        :rtype: numpy array/float
        """        
        return self.model.predict(X_test)