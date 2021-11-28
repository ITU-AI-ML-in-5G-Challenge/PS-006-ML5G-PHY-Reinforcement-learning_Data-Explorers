'''
UFPA - LASSE - Telecommunications, Automation and Electronics Research and Development Center - www.lasse.ufpa.br
CAVIAR - Communication Networks and Artificial Intelligence Immersed in Virtual or Augmented Reality
Ailton Oliveira, Felipe Bastos, João Borges, Emerson Oliveira, Daniel Suzuki, Lucas Matni, Rebecca Aben-Athar, Aldebaro Klautau (UFPA): aldebaro@ufpa.br
CAVIAR: https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY-RL.git

Enviroment for reinforcement learning applied to Beam-selection
V1.0
'''

import numpy as np
from gym import Env
from gym.spaces import Box, MultiDiscrete

from communications.buffer import Buffer
from communications.base_station import BaseStation
from communications.ue import UE
import matplotlib.pyplot as plt

class BeamSelectionEnv(Env):
    def __init__(self, ep=[0], steps_ep=200, state_used=['channel_mag', 'chosen_ue', 'pkts_transmitted'], length=3, low=[0, 0, 0], high=[2e4, 2, 1e3]):
        # Which episode to take data from (Only used when use_airsim=False).
        self.eps = ep
        '''
        Defining simulation environment with one BS and three UEs
        '''
        self.ue1 = UE(name='uav1', obj_type='UAV', total_number_rbs=15, episode = self.eps, use_airsim=False)
        self.ue2 = UE(name='simulation_car1', obj_type='CAR', total_number_rbs=15, episode = self.eps, use_airsim=False)
        self.ue3 = UE(name='simulation_pedestrian1', obj_type='PED', total_number_rbs=15, episode = self.eps, use_airsim=False)
        self.state_used = state_used
        self.length = length
        self.low = low
        self.high = high
        self.caviar_bs = BaseStation(Elements=64, frequency=60e9,name='BS1',ep_lenght=steps_ep, traffic_type = 'dense', BS_type = 'UPA', change_type=True, state_used=state_used)
        #Append users
        self.caviar_bs.append(self.ue1)
        self.caviar_bs.append(self.ue2)
        self.caviar_bs.append(self.ue3)
        self.rewards = np.zeros(steps_ep)
        self.sent_pkts_vec = np.zeros(steps_ep)
        self.dropped_pkts_vec = np.zeros(steps_ep)
        self.i = 0
        '''
        The observation space is composed by an array with 7 float numbers. 
        The first three represent the user position in XYZ, while the 
        remaining ones are respectively: dropped packages, sent packages, 
        buffered and bit rate.
        '''
        self.observation_space = Box(
            #low=np.array([-5e2,-5e2,-5e2,0,0,0,0]), 
            low=np.array(low), 
            high=np.array(high),
            #high=np.array([5e2,5e2,5e2,1e3,1e3,2e4,1e9]),
            shape=(length,)
    )
        '''
        The action space is composed by an array with two integers. The first one 
        represents the user that is currently being allocated and the second one, 
        the codebook index.
        '''
        self.action_space = MultiDiscrete([len(self.caviar_bs.UEs), self.caviar_bs._NTx])
        
        self.reset()


    def reset(self):
        self._state = np.zeros(self.length)
        done = False
        return self._state
    
    '''
    The step function receives a user and the beam index to serve it. The user state 
    is updated at every step by checking the correspondent element inside the simulator.
     
    :param action: (array) is composed by the user ID and the codebook index
    '''
    def step(self, action):
        target, index = action
        bs_example_state, bs_example_reward, info, done, n_steps, sent_pkts, dropped_pkts = self.caviar_bs.step(target,index)
        self.state = bs_example_state
        reward = bs_example_reward
        if done == True:
            x = range(n_steps)
            self.rewards[self.i] = reward + self.rewards[(self.i)-1]
            self.sent_pkts_vec[self.i] = sent_pkts
            self.dropped_pkts_vec[self.i] = dropped_pkts
            file = open("train_rewards_" + str(self.state_used[0:2]) + ".txt","w")
            for i in range(len(self.rewards)):
                file.write(str(x[i]) + "\t" + str(self.rewards[i]) + "\t" + str(self.sent_pkts_vec[i]) + "\t" + str(self.dropped_pkts_vec[i]) + "\n")
            #plt.plot(x, self.rewards)
            #plt.show()
        else:
            if self.i == 0:
                self.rewards[self.i] = reward
                self.sent_pkts_vec[self.i] = sent_pkts
                self.dropped_pkts_vec[self.i] = dropped_pkts
            else:
                self.rewards[self.i] = reward + self.rewards[(self.i)-1]
                self.sent_pkts_vec[self.i] = sent_pkts
                self.dropped_pkts_vec[self.i] = dropped_pkts
            self.i  = self.i  + 1
        #print(self.i)
        return self.state, reward, done, info
    
    def best_beam_step(self, target):
        bs_example_state, bs_example_reward, info, done = self.caviar_bs.best_beam_step(target)
        self.state = bs_example_state
        reward = bs_example_reward
        return self.state, reward, done, info