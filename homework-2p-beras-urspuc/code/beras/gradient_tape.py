from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """
        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.
        
        grads[id(target)] = [1]
        while queue:
            current_tensor = queue.pop(0)
            producer_layer = self.previous_layers[id(current_tensor)]

            if producer_layer is not None:
                input_tensors = producer_layer.inputs
                output_grad = grads[id(current_tensor)][0]

                input_grads = producer_layer.backward(output_grad)

                for input_tensor, grad in zip(input_tensors, input_grads):
                    if grads[id(input_tensor)] is None:
                        grads[id(input_tensor)] = [grad]
                    else:
                        grads[id(input_tensor)][0] += grad

                    queue.append(input_tensor)

        # while queue:
        #     current_tensor = queue.pop(0)
        #     current_id = id(current_tensor)
        #     if current_id in self.previous_layers:
        #         current_layer = self.previous_layers[current_id]
        #         current_grad = grads[current_id][0]  # Retrieve the current gradient

        #         # Compute gradients for layer inputs
        #         for input_tensor, grad in zip(current_layer.inputs, current_layer.compose_input_gradients(current_grad)):
        #             if grads[id(input_tensor)] is None:
        #                 grads[id(input_tensor)] = [grad]
        #             else:
        #                 grads[id(input_tensor)][0] += grad  # Accumulate gradients
        #             queue.append(input_tensor)

        #         # Compute gradients for layer weights
        #         for weight_tensor, grad in zip(current_layer.weights, current_layer.compose_weight_gradients(current_grad)):
        #             if grads[id(weight_tensor)] is None:
        #                 grads[id(weight_tensor)] = [grad]
        #             else:
        #                 grads[id(weight_tensor)][0] += grad  # Accumulate gradients
                   


        ## Retrieve the sources and make sure that all of the sources have been reached
        out_grads = [grads[id(source)][0] for source in sources]
        disconnected = [f"var{i}" for i, grad in enumerate(out_grads) if grad is None]

        if disconnected:
            print(f"Warning: The following tensors are disconnected from the target graph: {disconnected}")

        return out_grads