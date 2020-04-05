# Declarations
# Imports 
import random
import pandas as pd
import numpy as np

class Two_Sensors(object):
    def __init__(self, mu=0.1, m=2.5, g =9.8,v=1, max_reward = 1):
        """ This class contains the environment for the experiment II.

        :param mu: Friction coefficient between the box and the floor.
                   , defaults to 0.1
        :type mu: float, optional
        :param m: Mass of the bullet in Kg, defaults to 2.5
        :type m: float, optional
        :param g: Acceleration of gravity in m/s^2, defaults to 9.8
        :type g: float, optional
        :param v: Velocity of the bullet in m/s, defaults to 1
        :type v: int, optional
        :param max_reward: Maximum allowed reward for the agent, defaults to 1
        :type max_reward: int, optional
        """        

        self.max_reward = max_reward
        self.mu = mu
        self.m = m
        self.g = g
        self.v = v
        self.d_min = 0.8
        self.d_max = 8
        self.M_max = self.v*self.m*np.sqrt(0.5*(1/(self.g*self.mu*self.d_min)))-self.m
        self.M_min = self.v*self.m*np.sqrt(0.5*(1/(self.g*self.mu*self.d_max)))-self.m
        self.M = random.uniform(self.M_min, self.M_max)
        self.position_exact = 0.5 * (self.v*self.m/(self.M+self.m))**2 * (1/(self.g*self.mu))

        if self.position_exact>0 and self.position_exact<2:
            self.position_zone = 1
        elif self.position_exact>2 and self.position_exact<4:
            self.position_zone = 3
        elif self.position_exact>4 and self.position_exact<6:
            self.position_zone = 5
        elif self.position_exact>6 and self.position_exact<8:
            self.position_zone = 7
        self.total_reward = 0
        self.fail = 0

    def take_action(self, action):
        """ Given an action this function gives the corresponding outcome. The
        action 0 represents the low accuracy / wide range sensor. The actions 
        (1,2,3,4) represts the high accuracy / narrow range sensor and it's 
        placement. 
        
        :param action: Action representing the choice of sensor
        :type action: int
        :return: Outcome of the sensor
        :rtype: float / int
        """        
        if action == 0:
            outcome = self.position_zone
        elif action == 1:
            if self.position_zone == 1:
                outcome = self.position_exact
            else:
                outcome = -1
        elif action == 2:
            if self.position_zone == 3:
                outcome = self.position_exact
            else:
                outcome = -1
        elif action == 3:
            if self.position_zone == 5:
                outcome = self.position_exact
            else:
                outcome = -1
        elif action == 4:
            if self.position_zone == 7:
                outcome = self.position_exact
            else:
                outcome = -1
        return outcome
    
    def restart_mass(self):
        """Restart the mass M of the box and rewrites the variables with the
        the new positions.
        """        

        self.M = random.uniform(self.M_min, self.M_max)
        self.position_exact = 0.5 * (self.v*self.m/(self.M+self.m))**2 * (1/(self.g*self.mu))
        if self.position_exact > 0 and self.position_exact < 2:
            self.position_zone = 1
        elif self.position_exact > 2 and self.position_exact < 4:
            self.position_zone = 3
        elif self.position_exact > 4 and self.position_exact < 6:
            self.position_zone = 5
        elif self.position_exact > 6 and self.position_exact < 8:
            self.position_zone = 7

    def test_shooting(self):
        """ Samples uniformly at random a velocity for the bullet from the range
        [v-0.5*v,v+0.5*v], where v is the velocity of the bullets in the initial
        shooting. Then it calculates the position of the box after being shot 
        with the random velocity.

        :return: v_test (sampled velocity), d_test (corresponding distance)
        :rtype: float, float
        """        

        v_test = random.uniform(self.v-0.5*self.v, self.v+0.5*self.v) 
        d_test = 0.5*(v_test*self.m/(self.M+self.m))**2*1/(self.g*self.mu)
        return v_test, d_test

    def give_reward(self, prediction, observed, sigma=0.05):
        """ Generates the reward based on a Gaussian function of the relative
        difference between the predicted and the observed value.

        :param prediction: predicted value
        :type prediction: float
        :param observed: real observed value
        :type observed: float
        :param sigma: Width of the Gaussian. The smaller `sigma`, the higher the
                      needed precision to get a reward, defaults to 0.05
        :type sigma: float, optional
        :return: Obtained reward
        :rtype: float / numpy array
        """        
        x = (observed-prediction)/observed
        reward = self.max_reward*np.exp(-0.5*(x/sigma)**2)
        self.total_reward = self.total_reward + reward
        return reward

    def get_measurements(self, action1, outcome1, action2, outcome_2, V):
        """ Transforms the actions and the environmental elements into an array
        to feed the Analyzer.
        
        :param action1: Action performed by the first experimenter.
        :type action1: int
        :param outcome1: Outcome of the first use of the sensors.
        :type outcome1: float / int
        :param action2: Action performed by the second experimenter.
        :type action2: int
        :param outcome_2: Outcome of the second use of the sensors.
        :type outcome_2: float / int
        :param V: Velocity of the test bullet.
        :type V: float
        :return: Numpy array containing all the arguments
        :rtype: numpy array
        """        
        return np.array([action1, outcome1, action2, outcome_2, V])

    def reshape_for_analyzer(self, measurements, target):
        """ Reshapes the measurements and the target value in a suitable
        format for a Keras analyzer
        
        :param measurements: Measurements to give the Analyzer to make the 
                             prediction
        :type measurements: numpy array
        :param target: Target value to fit.
        :type target: float
        :return: X_train (inputs to the Analyzer), y_train (Target value to fit)
        :rtype: pandas DataFrame, pandas DataFrame
        """        
        X_train = pd.DataFrame(measurements.reshape(-1,len(measurements)))
        y_train = pd.DataFrame(target*np.ones((1,1)))
        return X_train, y_train
