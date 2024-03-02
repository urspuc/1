import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    """
    TODO: Implement for default intermediate activation.
        - call function
        - input gradients
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def call(self, x) -> Tensor:
        """TODO: Leaky ReLu forward propagation! """
        return np.where(x > 0, x, x*self.alpha)

    def get_input_gradients(self) -> list[Tensor]:
        """
        TODO: Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet!
        Make sure not to mutate any instance variables. Return a NEW list[tensor(s)]
        """
        x, y = self.inputs + self.outputs
        gradients = np.where(x > 0, 1, self.alpha)
        return [gradients]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    """
    TODO: Implement for default output activation to bind output to 0-1
        - call function
        - input_gradients 
    """ 
    
    def call(self, x) -> Tensor:
        return 1/( 1 + np.exp(-x))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet!
        Make sure not to mutate any instance variables. Return a NEW list[tensor(s)]
         """
        x, y = self.inputs + self.outputs
        return [y * (1-y)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    """
    TODO: Implement for default output activation to bind output to 0-1
        - call function
        - input_gradients
    """

    def call(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        x = x - np.max(x, axis = -1, keepdims = True)
        exps = np.exp(x)
        outs = exps / np.sum(exps, axis = -1, keepdims = True)
        return outs

    def get_input_gradients(self):
        """Softmax input gradients!"""
        # https://stackoverflow.com/questions/48633288/how-to-assign-elements-into-the-diagonal-of-a-3d-matrix-efficiently
        x, y = self.inputs + self.outputs
        batch_size, n = x.shape
        softmax_gradients = np.zeros(shape=(batch_size, n, n), dtype=x.dtype)
        for b in range(batch_size):
            out = y[b]
            softmax_gradients[b] = -np.outer(out, out)
            np.fill_diagonal(softmax_gradients[b], out * (1 - out))
        return [softmax_gradients]
