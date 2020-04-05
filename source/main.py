# Imports 
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from IPython.display import clear_output
import seaborn as sns
from rlagents import Q_Learning, DQN
from analyzers import Two_Layers_single_output as Analyzer
import environments
from keras.utils import to_categorical
sns.set()



N_Episodes = 200
count = 0
game = environments.Two_Sensors()
experimenter1 = Q_Learning(iterations=N_Episodes, output_size=5)
experimenter2 = DQN(iterations=N_Episodes, input_size=6, output_size=5)
analyzer = Analyzer(input_size=5)
t0 = time.time()
t1 = time.time()

# First random initialization of states:
action1 = experimenter1.get_next_action()
outcome1 = game.take_action(action1)
action2 = random.randint(0,4)
outcome2 = game.position_exact
d_test = 0.001
y_predicted = [0]
values = []
reward = 0
marker = 0
# Auxiliar variables for collecting data
df=pd.DataFrame(columns = ['Zone', 'Exact position', 'Action 1', 
                           'Outcome 1', 'Action 2', 'Outcome 2', 'Reward', 
                           'd_predicted', 'd_test', 'Exploration rate'])
total_reward_list = []


# Main loop
while count < N_Episodes:
    # Recording data
    values.append([game.position_zone, game.position_exact, 
                   action1, outcome1, action2, outcome2, float(reward), y_predicted[0], d_test, experimenter2.exploration_rate ])
    total_reward_list.append(game.total_reward)
   
    ### Sin esto ha funcionado!! (?¿???)
    old_state =  np.append(to_categorical(action1, 5), [outcome1])
    game.restart_mass()
    action1 = experimenter1.get_next_action()
    outcome1 = game.take_action(action1)
    current_state = np.append(to_categorical(action1, 5), [outcome1])
    experimenter2.online_train(old_state, action2, reward, current_state)
    action2 = experimenter2.get_next_action(state=current_state)
    outcome2 = game.take_action(action2)
    
    v_train, d_train = game.test_shooting()
    measurements = game.get_measurements(action1, outcome1, action2, outcome2, v_train)
#     analyzer.remember(d_train, measurements)
#     if marker > analyzer.batch:
#         marker = 0
#         analyzer.train()
#         print(count)
    
    X_train, y_train = game.reshape_for_analyzer(measurements, d_train) 
    analyzer.train(X_train, y_train)
    
    #Testing analyzer to generate reward
    v_test, d_test = game.test_shooting()
    measurements = game.get_measurements(action1,outcome1, action2, outcome2, v_test)
    X_test, y_test = game.reshape_for_analyzer(measurements, d_test)
    y_predicted = analyzer.predict(X_test)[0]
    reward = game.give_reward(y_predicted, d_test)
    
    # Update experimenter 1
    experimenter1.update(action1, reward)

# Display training status --------------------------------------
    count = count + 1
    marker = marker + 1
    if count % (N_Episodes/5000) == 0:
        clear_output()
        t2 = time.time()
        m, s = divmod(t2-t1, 60)
        mt, st = divmod(t2-t0, 60)
        me, se = divmod(((t2-t0)/count)*N_Episodes, 60)
        mr, sr = divmod(((t2-t0)/count)*N_Episodes-t2+t0, 60)
        print (str(int(count)) + '/' +  str(N_Episodes) + " episodes" + '(' + str(100*count/N_Episodes)+ '%)')
        print('Elapsed time: {} min {}s'.format(int(mt), int(st)))
        print("Est. completion time: {} min {}s,  Est. remaining time: {} min {}s".format(int(me), int(se), int(mr), int(sr)))
        t1 = time.time()
print('Training completed!')
plt.plot(range(len(total_reward_list)), total_reward_list)
plt.show()
# df=pd.DataFrame(columns = ['Zone', 'Exact position', 'Action 1', 
#                            'Outcome 1', 'Action 2', 'Outcome 2', 'Reward', 
#                            'd_predicted', 'd_test', 'Exploration rate'])
df=pd.DataFrame(values, columns = ['Zone', 'Exact position', 'Action 1', 
                                   'Outcome 1', 'Action 2', 'Outcome 2', 'Reward', 
                                   'd_predicted', 'd_test', 'Exploration rate'])
df.to_csv('df_nuevo.csv')