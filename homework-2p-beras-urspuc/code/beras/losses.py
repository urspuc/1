import numpy as np

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    """
    TODO:
        - call function
        - input_gradients
    Identical to HW1!
    """

    def call(self, y_pred: Tensor, y_true: Tensor) -> Tensor:

        MSE = np.mean((y_true - y_pred) **2)
        return Tensor(MSE)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Return in the order defined in the call function.
        """
        differences = self.inputs[1] - self.inputs[0]
        n,m = (self.inputs[0]).shape
        pred = -2 * differences / (n*m)

        return Tensor(pred), Tensor(np.zeros_like(self.inputs[1]))


def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1 - eps)


class CategoricalCrossentropy(Loss):
    """
    TODO: Implement CategoricalCrossentropy class
        - call function
        - input_gradients

        Hint: Use clip_0_1 to stabilize calculations
    """

    def call(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred_clipped = clip_0_1(y_pred)
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return Tensor(np.mean(loss))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        y_pred_clipped = clip_0_1(y_pred)
        input_gradients = np.where(y_true, -1/y_pred_clipped, 1) / y_pred.shape[0]

        return [input_gradients, -input_gradients]