import pytest
import numpy as np
from mdsdl.utilities import tanh_function, tanh_derivative
from mdsdl.fully_connected import FullyConnectedLayer

class TestFullyConnectedLayer:
    def test_fully_connected_layer_initialization(self):
        n_inputs = 3
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs, seed=42)

        assert layer.weights.shape == (n_inputs, n_outputs), "Weights shape mismatch"
        assert layer.biases.shape == (1, n_outputs), "Biases shape mismatch"
        assert np.all(layer.weights >= -0.5) and np.all(layer.weights <= 0.5), "Weights not in expected range"
        assert np.all(layer.biases >= -0.5) and np.all(layer.biases <= 0.5), "Biases not in expected range"

