import tensorflow as tf
from tensorflow.keras import optimizers
from MyModel import MyModel
import numpy as np

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr) #optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences 

    def get_action(self, states, epsilon):
        action_choice = [3,64]
        if np.random.random() < epsilon:
            #print(np.random.choice(self.num_actions))
            return [np.random.randint(action_choice[0]),np.random.randint(action_choice[1])]
        else:
            #print("input= ", np.atleast_2d(states))
            #print("output= ", self.predict(np.atleast_2d(states)))
            t = self.predict(np.atleast_2d(states))
            t_reshaped = tf.reshape(t, [3, 64])
            return np.unravel_index(np.argmax(t_reshaped, axis=None), t_reshaped.shape) #return the index of max Q value

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        #print((self.experience['a'][ids[0]])[0])
        #print((self.experience['a'][ids[0]])[1])
        actions = np.asarray([((self.experience['a'][i])[0]*64 + (self.experience['a'][i])[1]) for i in ids])
        #print(actions)
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        #print(states_next.shape)
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        #print(TargetNet.predict(states_next))
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)
    
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def predict_test(self, inputs):
        t = self.model(np.atleast_2d(inputs.astype('float32')))
        t_reshaped = tf.reshape(t, [3, 64])
        return np.unravel_index(np.argmax(t_reshaped, axis=None), t_reshaped.shape), 0 #return the index of max Q value
        
    def save_final_model(self):
        model_path = "./model/test.model"
        self.model.save(model_path) 