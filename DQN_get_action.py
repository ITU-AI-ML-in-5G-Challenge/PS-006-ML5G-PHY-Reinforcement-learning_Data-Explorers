def get_action(self, states, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(self.num_actions)
    else:
        return np.argmax(self.predict(np.atleast_2d(states))[0])