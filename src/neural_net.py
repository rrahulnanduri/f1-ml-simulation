import numpy as np

class NeuralNetwork:
    """
    A simple Feed-Forward Neural Network for driving.
    Architecture: 5 Inputs -> 16 Hidden (Tanh) -> 2 Outputs (Tanh)
    """
    def __init__(self, input_size=5, hidden_size=16, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases randomly between -1 and 1
        self.w1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, self.hidden_size))
        
        self.w2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(-1, 1, (1, self.output_size))

    def feed_forward(self, inputs):
        """
        Calculates outputs based on input sensor data.
        inputs: List or array of 5 floats (normalized distances)
        returns: (steering, throttle)
        """
        # Convert inputs to a 2D array if they aren't already
        x = np.array(inputs).reshape(1, -1)
        
        # Input to Hidden
        h1 = np.tanh(np.dot(x, self.w1) + self.b1)
        
        # Hidden to Output
        output = np.tanh(np.dot(h1, self.w2) + self.b2)
        
        # We use tanh to get outputs between -1 and 1
        # output[0][0] = steering
        # output[0][1] = throttle
        return output[0, 0], output[0, 1]

    def get_weights(self):
        """Returns the network weights for saving/evolution."""
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy()
        }

    def set_weights(self, weights):
        """Sets the network weights (used during evolution)."""
        self.w1 = weights["w1"].copy()
        self.b1 = weights["b1"].copy()
        self.w2 = weights["w2"].copy()
        self.b2 = weights["b2"].copy()
