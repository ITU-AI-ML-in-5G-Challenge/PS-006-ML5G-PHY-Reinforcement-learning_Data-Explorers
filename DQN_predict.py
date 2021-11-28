    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32'))) 