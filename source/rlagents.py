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

class Q_Learning(object):
    def __init__(self, iterations, output_size, learning_rate=0.1, discount=0,
    exploration_rate=1):
        """Agent based on the classic Q-Learning algorithm with no inputs.
        
        :param iterations: Number of iterations for the training. This parameter
                           is used to calculate the decrease on the exploration 
                           rate.
        :type iterations: int
        :param output_size: Number of actions available for the agent.
        :type output_size: int
        :param learning_rate: Learning rate for Q-Learning update rule, defaults
                              to 0.1
        :type learning_rate: float, optional
        :param discount: Discount factor. If it's 0 only cares about 
                         immediate reward. The closer to one the more 
                         it values future rewards, defaults to 0
        :type discount: int, optional
        :param exploration_rate: Initial exploration rate for the epsilon-greedy
                                 decision algorithm, defaults to 1
        :type exploration_rate: int, optional
        """

        self.output_size = output_size
        self.q_table = [0] * self.output_size # Spreadsheet (Q-table) 
        # for rewards 
        # accounting, 5 possible actions
        self.learning_rate = learning_rate # How much we appreciate the new 
        # -value over current
        self.discount = discount # How much we appreciate future 
        # reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations # Shift 
        # from exploration to exploration        
    def get_next_action(self):
        """Uses the model and the exploration rate to choose a an action with an
        epsilon-greedy decision algorithm.
        
        :return: Integer representing the action
        :rtype: int
        """        
        if random.random() > self.exploration_rate:
            return self.greedy_action()
        else:
            return self.random_action()
    def greedy_action(self):
        """ Takes the index of the maximum Q-Value to choose an action.
        
        :return: Integer representing the model action
        :rtype: int
        """        
        return np.argmax(self.q_table)
    def random_action(self):
        """Generates a random integer to choose a random action.
        
        :return: Integer representing the random action
        :rtype: int
        """        
        return random.randint(0,self.output_size-1)
    def update(self, action, reward):
        """ Takes the last action taken and the reward obtained to update the 
        Q-Value of the action by using the Bellman equation.
        
        :param action: Last action taken by the agent
        :type action: int
        :param reward: Reward obtained with the taken action
        :type reward: float
        """        
        old_value = self.q_table[action]
        # What would be our best next action?
        future_action = self.greedy_action()
        # What is reward for th best next action?
        future_reward = self.q_table[future_action]
        # Main Q-table updating algorithm
        new_value = old_value + self.learning_rate * (reward + 
        self.discount*future_reward - old_value)
        self.q_table[action] = new_value
        #Each train we update the exploration rate.
        if self.exploration_rate > 0:
            self.exploration_rate = self.exploration_rate-2*self.exploration_delta

class DQN(object):
    def __init__(self, iterations, input_size, output_size, learning_rate=0.01,
    discount=0, exploration_rate=1):
        """ Agent based on a simple Deep Q-Network with inputs and outputs. It
        uses online learning (no replay memory) to train the train a simple
        feedforward network

        :param iterations: Number of iterations for the training. This parameter
                           is used to calculate the decrease on the exploration 
                           rate.
        :type iterations: int
        :param input_size: Number of features given to the agent
        :type input_size: int
        :param output_size: Number of actions available for the agent.
        :type output_size: int
        :param learning_rate: Learning rate for Q-Learning update rule, 
                              defaults to 0.01
        :type learning_rate: float, optional
        :param discount: Discount factor. If it's 0 only cares about 
                         immediate reward. The closer to one the more it values 
                         future rewards, defaults to 0, defaults to 0
        :type discount: int, optional
        :param exploration_rate: Initial exploration rate for the epsilon-greedy
                                 decision algorithm, defaults to 1, defaults to 1
        :type exploration_rate: int, optional
        """
        
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.exploration_delta = 1.0 / iterations
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_model()

    def build_model(self):
        """Uses keras to build de model. You should modify here the code to
        change the neural network architecture. Default to a 
        *input_sizex64x64xoutput_size* fully connected sequential network.
        
        :return: Keras model for the agent using the specified structure. 
        :rtype: keras model
        """        
        input_data = Input(shape=(self.input_size,))
        layer1 = Dense(64, activation='sigmoid')(input_data)
        layer2 = Dense(64, activation='sigmoid')(layer1)
        output_data = Dense(self.output_size)(layer2)
        model = Model(input_data, output_data)
        model.compile(optimizer='adam', loss = 'mean_squared_error')
        return model
        
    def flat(self, state):
        """Reshapes the state to a flat numpy array
        
        :param state: Input state
        :type state: numpy array
        :return: Flat input state
        :rtype: numpy array
        """

        flat = state.reshape((1,self.input_size))
        return flat
    
    def get_Q(self, state):
        """Calculate the Q-Values for all the actions using the trained Deep 
        Q-Network.
        :param state: Input state for the agent.
        :type state: numpy array
        :return: A ordered table with the Q-Value of each action for the given
                 state.
        :rtype: numpy array? (I think)
        """        
        return self.model.predict(state)
    
    def random_action(self):
        """Generates a random integer to choose a random action.
        
        :return: Integer representing the random action
        :rtype: int
        """
        return random.randint(0,self.output_size-1)

    def greedy_action(self, state):
        """ Takes the index of the maximum Q-Value to choose an action.
        
        :return: Integer representing the model action
        :rtype: int
        """        

        state_flat = self.flat(state) 
        return np.argmax(self.get_Q(state_flat))
    
    def get_next_action(self, state):
        """Uses the model and the exploration rate to choose a an action with an
        epsilon-greedy decision algorithm.
        
        :return: Integer representing the action
        :rtype: int
        """        

        if self.exploration_rate>0:
            self.exploration_rate = self.exploration_rate-2*self.exploration_delta
        if random.random() > self.exploration_rate:
            return self.greedy_action(state)
        else:
            return self.random_action()

    def online_train(self, old_state, old_action, reward, current_state):
        """ Uses keras built in functions to estimate the values of the 
        previous state and the future state to calculate the temporal
        difference and use it to fit the model.
        
        :param old_state: state from which the action was taken.
        :type old_state: numpy array?
        :param old_action: action that was taken
        :type old_action: int
        :param reward: reward obtained from the result of old_action
        :type reward: float
        :param current_state: state after the action old_action was taken
        :type current_state: numpy array?
        """        
        old_state_flat = self.flat(old_state)
        current_state_flat = self.flat(current_state)
        target = self.model.predict(old_state_flat)
        Q_future = max(self.model.predict(current_state_flat)[0])
        target[0][old_action] = reward + Q_future*self.discount
        self.model.fit(old_state_flat, target, epochs=1, verbose=0)