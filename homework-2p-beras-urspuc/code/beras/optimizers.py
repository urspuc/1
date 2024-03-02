from collections import defaultdict
import numpy as np

"""
TODO: Implement all the apply_gradients for the 3 optimizers:
    - BasicOptimizer
    - RMSProp
    - Adam
"""

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, weights, grads):
        for i in range( len(weights)):
            if weights[i].trainable == True:
                weights[i] -= self.learning_rate * grads[i]

class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, weights, grads):
        ## TODO: Implement RMSProp optimization
        ## HINT: Lab 2?
        for i, grad in enumerate(grads):
            if weights[i].trainable:
                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad ** 2)
                weights[i] -= self.learning_rate * grad / (np.sqrt(self.v[i]) + self.epsilon)

class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.amsgrad = amsgrad

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.m_hat = defaultdict(lambda: 0)     # Expected value of first moment vector
        self.v_hat = defaultdict(lambda: 0)     # Expected value of second moment vector
        self.t = 0                              # Time counter

    def apply_gradients(self, weights, grads):
        ## TODO: Implement Adam optimization
        ## HINT: Lab 2?
        self.t += 1 # Increment time step
        for i, grad in enumerate(grads):
            if weights[i].trainable:
                self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad
                self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (grad ** 2)

                m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

                if self.amsgrad:
                    self.v_hat[i] = np.maximum(self.v_hat[i], v_hat)
                    v_hat = self.v_hat[i]

                weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

