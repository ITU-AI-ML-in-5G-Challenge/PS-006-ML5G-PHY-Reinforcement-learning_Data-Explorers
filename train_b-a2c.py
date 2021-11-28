'''
UFPA - LASSE - Telecommunications, Automation and Electronics Research and Development Center - www.lasse.ufpa.br
CAVIAR - Communication Networks and Artificial Intelligence Immersed in Virtual or Augmented Reality
Ailton Oliveira, Felipe Bastos, João Borges, Emerson Oliveira, Daniel Suzuki, Lucas Matni, Rebecca Aben-Athar, Aldebaro Klautau (UFPA): aldebaro@ufpa.br
CAVIAR: https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY-RL.git

Script to train the baseline of reinforcement learning applied to Beam-selection
V1.0
'''

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import caviar_tools
from beamselect_env import BeamSelectionEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import A2C
from Myplot import Myplot
#from stable_baselines3 import A2C


# Create the folder
try:
    os.mkdir('./model')
except OSError as error:
    print(error)

'''
Trains an A2C network and stores it in a file.

Usage:

$ python3 train_b-a2c.py -m <model_name> -ep <train_ep_id#first> <train_ep_id#last>

Example:

$ python3 train_b-a2c.py -m baseline.a2c -ep 0 1
'''
parser = argparse.ArgumentParser()

parser.add_argument("--model", "-m", 
                    help="Pass RL model name",
                    action="store", 
                    dest="model", 
                    type=str)

parser.add_argument("--episode", "-ep",
                    nargs='+',
                    help="IDs of the first and " +
                         "last episodes to train", 
                    action="store", 
                    dest="episode", 
                    type=str)
                   
args = parser.parse_args()

# Get total number of steps based on the timestamps for a specific UE  
n_steps = caviar_tools.linecount(args.episode)
print(n_steps)

#states = np.array([['pos_x','pos_y','pos_z','pkts_dropped','pkts_transmitted','pkts_buffered','bit_rate']])
#lows = np.array([[-50, -50, -50, 0, 0, 0, 0]])
#highs = np.array([[50, 50, 50, 1e3, 2e4, 1e3, 1e9]])

#states = np.array([['channel_mag', 'target', 'pkts_transmitted'], ['channel_mag'], ['channel_mag','pos_x','pos_y','pos_z'], ['pkts_transmitted', 'pkts_buffered', 'pkts_dropped'], ['pkts_transmitted']])
#lows = np.array([[0, 0, 0], [0], [0, -50, -50, -50], [0, 0, 0], [0]])
#highs = np.array([[2e4, 2, 1e3], [2e4], [2e4, 50, 50, 50], [1e3, 2e4, 1e3], [1e3]])

#states = np.array([['pos_x','pos_y','pos_z'], ['pkts_buffered'], ['pos_x','pos_y','pos_z','position_other1_x', 'position_other1_y', 'position_other1_z', 'position_other2_x', 'position_other2_y', 'position_other2_z'], ['buffered_packets_target', 'buffered_packets_other1', 'buffered_packets_other2'], ['target', 'buffered_packets_target', 'buffered_packets_other1', 'buffered_packets_other2']])
#lows = np.array([[-50, -50, -50], [0], [-50, -50, -50,-50, -50, -50,-50, -50, -50], [0,0,0], [0,0,0,0]])
#highs = np.array([[50, 50, 50], [2e4], [50, 50, 50,50, 50, 50,50, 50, 50], [2e4,2e4,2e4], [2,2e4,2e4,2e4]])

#states = np.array([['channel_mag', 'target'], ['pos_x','pos_y','pos_z','position_other1_x', 'position_other1_y', 'position_other1_z', 'position_other2_x', 'position_other2_y', 'position_other2_z', 'buffered_packets_target', 'buffered_packets_other1', 'buffered_packets_other2'], ['channel_mag', 'target', 'buffered_packets_target', 'buffered_packets_other1', 'buffered_packets_other2']])
#lows = np.array([[0, 0], [-50, -50, -50,-50, -50, -50,-50, -50, -50, 0,0,0], [0,0,0,0,0]])
#highs = np.array([[2e4, 2], [50, 50, 50,50, 50, 50,50, 50, 50, 2e4,2e4,2e4], [2e4,2,2e4,2e4,2e4]])

#states = np.array([['channel_mag', 'target', 'pkts_transmitted'], ['channel_mag','pos_x','pos_y','pos_z'], ['pkts_transmitted', 'pkts_buffered', 'pkts_dropped'], ['pos_x','pos_y','pos_z','position_other1_x', 'position_other1_y', 'position_other1_z', 'position_other2_x', 'position_other2_y', 'position_other2_z', 'buffered_packets_target', 'buffered_packets_other1', 'buffered_packets_other2'], ['target', 'buffered_packets_target', 'buffered_packets_other1', 'buffered_packets_other2']])
#lows = np.array([[0, 0, 0], [0, -50, -50, -50], [0, 0, 0], [-50, -50, -50,-50, -50, -50,-50, -50, -50, 0,0,0], [0,0,0,0]])
#highs = np.array([[2e4, 2, 1e3], [2e4, 50, 50, 50], [1e3, 2e4, 1e3], [50, 50, 50,50, 50, 50,50, 50, 50, 2e4,2e4,2e4], [2,2e4,2e4,2e4]])

#specify the states you have with their max and min values as shown above examples

states = np.array([['target','buffered_packets_target','buffered_packets_other1','buffered_packets_other2']])
lows = np.array([[0,0,0,0]])
highs = np.array([[2,2e4,2e4,2e4]])


for i in range(states.shape[0]):

    state_used = states[i]
    length = len(state_used)
    low = lows[i]
    high = highs[i]
    e = BeamSelectionEnv(ep=args.episode, steps_ep=n_steps, state_used=state_used, length=length, low=low, high=high)
    
    model = A2C(policy="MlpPolicy", 
                learning_rate=1e-3, 
                n_steps=2, 
                verbose=1,
                gamma=0.99, 
                env=e, 
                seed=0,
                tensorboard_log="./log_tensorboard/")
    
    model.learn(total_timesteps=n_steps)
    model_path = "./model/"+str(args.model)
    model.save(model_path)

file_list = []
for i in range(states.shape[0]):
    file_list.append('train_rewards_' + str(states[i][0:2]) + '.txt')

Myplot(file_list)