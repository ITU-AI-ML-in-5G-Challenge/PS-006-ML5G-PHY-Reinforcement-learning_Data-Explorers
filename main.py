import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers
from DQN_init import DQN
from play_game import play_game
from test_b_a2c import test_beam_selection

def main(env,N,count_per_episode):
    #env = gym.make('CartPole-v0')
    gamma = 0.99
    if len(env.eps)<2:
        Num_of_episodes = 1
    else:
        Num_of_episodes = len(range(int(env.eps[0]), int(env.eps[1])+1))
    k = count_per_episode #vector stores steps of each episode
    copy_step = 25
    num_actions = 3*64
    num_states = len(env.observation_space.sample())
    #num_actions[0] = 64 #len(env.action_space[0].n)
    #num_actions[1] = 3 #len(env.action_space[1].n)
    #print("action #: ",num_actions)
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-4
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    #summary_writer = tf.summary.FileWriter(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    #N = 200
    total_rewards = np.empty(Num_of_episodes)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(Num_of_episodes):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step, k[n])
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        #if n % 100 == 0:
        print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("avg reward for last 100 episodes:", avg_rewards)
    test_beam_selection(TrainNet,"TrainNet",[400,450],True)
    #TrainNet.save_final_model()
    #make_video(env, TrainNet)
    env.close()