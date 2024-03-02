from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING, TypedDict, Dict, Union, Any, Optional, List, Iterable

if TYPE_CHECKING:
    from beras.gradient_tape import GradientTape


class Tensor(np.ndarray):
    """
    Essentially, a NumPy Array that can also be marked as trainable
    Custom Tensor class that mimics tf.Tensor. Allows the ability for a numpy array to be marked as trainable.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.trainable = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.trainable = getattr(obj, "trainable", True)


"""
Mimics the tf.Variable class.
"""
Variable = Tensor


class Callable(ABC):
    """
    Modules that can be called like functions.
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Ensures `self()` and `self.call()` be the same
        """
        return Tensor(self.call(*args, **kwargs))

    @abstractmethod
    def call(self, *args, **kwargs) -> Tensor:
        """
        Pass inputs through function.
        """
        pass


class Weighted(ABC):
    """
    Modules that have weights.
    """

    @property
    @abstractmethod
    def weights(self) -> list[Tensor]:
        pass

    @property
    def trainable_variables(self) -> list[Tensor]:
        return [w for w in self.weights if w.trainable]

    @property
    def non_trainable_variables(self) -> list[Tensor]:
        return [w for w in self.weights if not w.trainable]

    @property
    def trainable(self) -> bool:
        return len(self.trainable_variables) > 0

    @trainable.setter
    def trainable(self, value: bool):
        for w in self.trainable_variables:
            w.trainable = value


class Diffable(Callable, Weighted):
    """
    Modules that keep track of gradients
    """

    # Shared across all Diffable instances
    gradient_tape: GradientTape | None = None

    def __call__(self, *args, **kwargs) -> Tensor | list[Tensor]:
        """
        If there is a gradient tape scope in effect, perform AND RECORD the operation.
        Otherwise... just perform the operation and don't let the gradient tape know.
        The call function for a Diffable object typically performs a forward pass in
        a neural network, while keeping track of inputs, outputs, arguments to the layer,
        and (if applicable) gradient tape records.
        """

        # The call method keeps track of method inputs and outputs
        self.argnames = self.call.__code__.co_varnames[1:]
        named_args = {self.argnames[i]: args[i] for i in range(len(args))}
        self.input_dict = {**named_args, **kwargs}
        self.inputs = [
            self.input_dict[arg]
            for arg in self.argnames
            if arg in self.input_dict.keys()
        ]
        self.outputs = self.call(*args, **kwargs)

        ## Make sure outputs are tensors and tie back to this layer
        list_outs = isinstance(self.outputs, list) or isinstance(self.outputs, tuple)
        if not list_outs:
            self.outputs = [self.outputs]

        ## When the Diffable is a part of an active gradient tape scope, record the operation
        if Diffable.gradient_tape is not None:
            for out in self.outputs:
                Diffable.gradient_tape.previous_layers[id(out)] = self

        ## And then finally, it returns the output, thereby wrapping the forward call
        return self.outputs if list_outs else self.outputs[0]

    @abstractmethod
    def get_input_gradients(self) -> list[Tensor]:
        """
        NOTE: required for all Diffable modules
        returns:
            list of gradients with respect to the inputs
        """
        return []

    @abstractmethod
    def get_weight_gradients(self) -> list[Tensor]:
        """
        NOTE: required for SOME Diffable modules
        returns:
            list of gradients with respect to the weights
        """
        return []

    def compose_input_gradients(self, J: Iterable = None):
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied

        Note: This and compose_to_weight are generalized to handle a wide array of architectures
                so it handles a lot of edge cases that you may not need to worry about for this
                assignment. That being said, it's very close to how this really works in Tensorflow and
                it's helps A LOT to understand how this works so you can debug the gradient method.
        """

        # Returns input gradients if no apriori cumulative jacobians are provided (i.e. loss layer).
        if J is None or J[0] is None:
            return self.get_input_gradients()
        # J_out stores all input gradients to be tracked in backpropagation.
        J_out = []
        for upstream_jacobian in J:
            batch_size = upstream_jacobian.shape[0]
            for layer_input, inp_grad in zip(self.inputs, self.get_input_gradients()):
                j_wrt_lay_inp = np.zeros(layer_input.shape, dtype=inp_grad.dtype)
                for sample in range(batch_size):
                    s_grad = inp_grad[sample] if len(inp_grad.shape) == 3 else inp_grad
                    try:
                        j_wrt_lay_inp[sample] = s_grad @ upstream_jacobian[sample]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b @ j[b]` with {s_grad.shape} and {upstream_jacobian[sample].shape}:\n{e}"
                        )
                J_out += [j_wrt_lay_inp]
        # Returns cumulative jacobians w.r.t to all inputs.
        return J_out

    def compose_weight_gradients(self, J: Iterable = None) -> list[Tensor]:
        """
        Compose the inputted cumulative jacobian with the weight jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `weight_gradients` to provide either batched or overall jacobian.
        Assumes weight/cumulative jacobians are element-wise multiplied (w/ broadcasting)
        and the resulting per-batch statistics are averaged together for avg per-param gradient.
        """
        # Returns weights gradients if no apriori cumulative jacobians are provided.
        if J is None or J[0] is None:
            return self.get_weight_gradients()
        # J_out stores all weight gradients to be tracked in further backpropagation.
        J_out = []
        ## For every weight/weight-gradient pair...
        for upstream_jacobian in J:
            for layer_w, w_grad in zip(self.weights, self.get_weight_gradients()):
                batch_size = upstream_jacobian.shape[0]
                ## Make a cumulative jacobian which will contribute to the final jacobian
                j_wrt_lay_w = np.zeros((batch_size, *layer_w.shape), dtype=w_grad.dtype)
                ## For every element in the batch (for a single batch-level gradient updates)
                for sample in range(batch_size):
                    ## If the weight gradient is a batch of transform matrices, get the right entry.
                    ## Allows gradient methods to give either batched or non-batched matrices
                    s_grad = w_grad[sample] if len(w_grad.shape) == 3 else w_grad
                    ## Update the batch's Jacobian update contribution
                    try:
                        j_wrt_lay_w[sample] = s_grad * upstream_jacobian[sample]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b * j[b]` with {s_grad.shape} and {upstream_jacobian[sample].shape}:\n{e}"
                        )
                ## The final jacobian for this weight is the average gradient update for the batch
                J_out += [np.sum(j_wrt_lay_w, axis=0)]
            ## After new jacobian is computed for each weight set, return the list of gradient updatates
        return J_out
