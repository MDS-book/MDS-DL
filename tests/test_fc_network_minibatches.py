import pytest
import numpy as np
from numpy import ndarray
from mdsdl.fully_connected import FullyConnectedLayer, ActivationLayer
from mdsdl.fully_connected.fc_network_minibatches import FCNetwork as FCNetworkMinibatches
from mdsdl.fully_connected.fc_network import FCNetwork
from mdsdl.utilities import tanh_function, tanh_derivative, \
    sigmoidal_function, sigmoidal_derivative, \
    MSE, MSE_derivative
from tests.reference_network_with_two_units import ReferenceNetworkWith2Units


# def test_minibatch_training():
#     X_train = np.array([[0.5, -1.0],
#                         [0.2, 0.1],
#                         [-0.2, 0.5]])
#     Y_train = np.array([[0.7],
#                         [-0.3],
#                         [0.1]])
#
#     # Initialize network
#     network = FCNetworkMinibatches(MSE, MSE_derivative)
#     network.add_layer(FullyConnectedLayer(2, 3, seed=42))
#     network.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
#     network.add_layer(FullyConnectedLayer(3, 1, seed=42))
#     network.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
#
#     # Train with mini-batches
#     costs = network.train(X_train, Y_train, epochs=100, learning_rate=0.1, batch_size=2)
#
#     # Plot training cost
#     import matplotlib.pyplot as plt
#     plt.plot(costs)
#     plt.xlabel('Epochs')
#     plt.ylabel('Cost')
#     plt.title('Training Cost Over Epochs')
#     plt.show()





@pytest.fixture
def network():
    """Fixture to initialize a sample network."""
    net = FCNetworkMinibatches(MSE, MSE_derivative)
    net.add_layer(FullyConnectedLayer(2, 3, seed=42))
    net.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
    net.add_layer(FullyConnectedLayer(3, 1, seed=42))
    net.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
    return net


@pytest.fixture
def data():
    """Fixture to provide sample training and testing data."""
    X_train = np.array([[0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6],
                        [0.7, 0.8]])
    Y_train = np.array([[0.3],
                        [0.7],
                        [1.1],
                        [1.5]])
    X_test = np.array([[0.9, 1.0]])
    Y_test = np.array([[1.9]])
    return X_train, Y_train, X_test, Y_test


@pytest.mark.parametrize("batch_size", [1, 2, 4])  # Test SGD, mini-batch, and full-batch training
def test_training_convergence(network, data, batch_size):
    """Test that the cost decreases over epochs for various batch sizes."""
    X_train, Y_train, _, _ = data
    epochs = 50
    learning_rate = 0.01

    # Train the network
    costs = network.train(X_train, Y_train, epochs, learning_rate, batch_size)

    # Assert the cost decreases
    assert costs[0] > costs[-1], f"Cost did not decrease with batch size {batch_size}"

def test_prediction_shape(network, data):
    """Test that the prediction shape matches the target shape."""
    _, _, X_test, Y_test = data

    # Predict with the network
    predictions = network.predict(X_test)

    # Assert shape consistency
    assert predictions.shape == Y_test.shape, "Prediction shape does not match target shape"

def test_shape_consistency_in_training(network, data):
    """Ensure shape consistency during forward and backward propagation."""
    X_train, Y_train, _, _ = data
    batch_size = 2
    learning_rate = 0.01

    # Simulate one batch
    X_batch = X_train[:batch_size]
    Y_batch = Y_train[:batch_size]

    # Forward propagation
    y_pred = X_batch
    for layer in network.layers:
        y_pred = layer.feed_forward(y_pred)

    assert y_pred.shape == (batch_size, Y_train.shape[1]), \
        "Output shape mismatch during forward propagation"

    # Backward propagation
    error = MSE_derivative(Y_batch, y_pred)
    for layer in reversed(network.layers):
        error, dJdW, dJdb = layer.backward_propagation(error, learning_rate)

        # Check gradient shapes
        if dJdW is not None:
            assert dJdW.shape == layer.weights.shape, "Weight gradient shape mismatch"
        if dJdb is not None:
            assert dJdb.shape == layer.biases.shape, "Bias gradient shape mismatch"


def test_batch_gradient_descent_with_reference__preparation():
    # Initialize weights and biases for the reference network
    W_ref = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    B_ref = (0.1, 0.2, 0.3)

    # Training data (X1, X2 as inputs, Y as target)
    X1 = np.array([0.1, 0.2, 0.3])
    X2 = np.array([0.4, 0.5, 0.6])
    Y = np.array([0.7, 0.8, 0.9])

    # Learning rate
    learning_rate = 0.01

    # Initialize an FCNetwork matching the reference network's structure
    network = FCNetworkMinibatches(MSE, MSE_derivative)
    network.add_layer(FullyConnectedLayer(2, 2, seed=42))  # Input to 2 hidden neurons
    network.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
    network.add_layer(FullyConnectedLayer(2, 1, seed=42))  # Hidden to output neuron

    # Manually set weights and biases to match the reference implementation
    network.layers[0].weights = np.array([[0.1, 0.3],
                                          [0.2, 0.4]])  # W_ref[:4] reshaped
    network.layers[0].biases = np.array([[0.1, 0.2]])  # B_ref[:2]
    network.layers[2].weights = np.array([[0.5],
                                          [0.6]])  # W_ref[4:]
    network.layers[2].biases = np.array([[0.3]])  # B_ref[2]

    # Reference outputs
    a1_ref, a2_ref, y1_ref, y2_ref, a3_ref = ReferenceNetworkWith2Units.forward_pass(
        W=W_ref, B=B_ref, X1=X1, X2=X2
    )

    # FCNetwork outputs
    X_batch = np.column_stack((X1, X2))
    y_pred_fc = X_batch
    for layer in network.layers:
        y_pred_fc = layer.feed_forward(y_pred_fc)

    # Compare final predictions
    np.testing.assert_allclose(y_pred_fc, a3_ref.reshape(-1, 1)), "Predictions do not match between FCNetwork and reference"
    ###################################


def test_batch_gradient_descent_with_reference():
    """Test that the batch gradient descent updates in FCNetwork match the reference implementation."""
    # Initialize weights and biases for the reference network
    W_ref = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    B_ref = (0.1, 0.2, 0.3)

    # Training data (X1, X2 as inputs, Y as target)
    X1 = np.array([0.1, 0.2, 0.3])
    X2 = np.array([0.4, 0.5, 0.6])
    Y = np.array([0.7, 0.8, 0.9])

    # Learning rate
    learning_rate = 0.01

    # Reference implementation: Compute weight and bias updates
    dJdW_ref, dJdB_ref = ReferenceNetworkWith2Units.backward_pass(W_ref, B_ref, X1, X2, Y)

    # Initialize an FCNetwork matching the reference network's structure
    network = FCNetworkMinibatches(MSE, MSE_derivative)
    network.add_layer(FullyConnectedLayer(2, 2, seed=42))  # Input to 2 hidden neurons
    network.add_layer(ActivationLayer(sigmoidal_function, sigmoidal_derivative))
    network.add_layer(FullyConnectedLayer(2, 1, seed=42))  # Hidden to output neuron

    # Manually set weights and biases to match the reference implementation
    network.layers[0].weights = np.array([[0.1, 0.3],
                                          [0.2, 0.4]])  # W_ref[:4] reshaped
    network.layers[0].biases = np.array([[0.1, 0.2]])               # B_ref[:2]
    network.layers[2].weights = np.array([[0.5], [0.6]])           # W_ref[4:]
    network.layers[2].biases = np.array([[0.3]])                   # B_ref[2]



    # Forward pass with the batch
    X_batch = np.column_stack((X1, X2))  # Combine X1 and X2 into a 2D array
    Y_batch = Y.reshape(-1, 1)           # Reshape Y into a column vector

    # Perform one training step (manual backpropagation) in the FCNetwork
    y_pred = X_batch
    for layer in network.layers:
        y_pred = layer.feed_forward(y_pred)

    # Backpropagation
    dJdW_fc = []
    dJdB_fc = []
    error = MSE_derivative(Y_batch, y_pred)
    for layer in reversed(network.layers):
        error, dJdW, dJdB = layer.backward_propagation(error, learning_rate)
        if isinstance(layer, FullyConnectedLayer):
            dJdW_fc.append(dJdW)
            dJdB_fc.append(dJdB)

    # Reverse to match layer order (input to output)
    dJdW_fc = dJdW_fc[::-1]
    dJdB_fc = dJdB_fc[::-1]

    print("\ndJdW_fc:\n", dJdW_fc)
    print("\ndJdW_ref:\n", dJdW_ref)

    # Aggregate hidden layer weights
    hidden_layer_weights = np.array([
        [np.mean(dJdW_ref[0]), np.mean(dJdW_ref[2])],
        [np.mean(dJdW_ref[1]), np.mean(dJdW_ref[3])],
    ])
    print("\nnp.mean(dJdW_ref[3])\n:", np.mean(dJdW_ref[3]))

    # Aggregate output layer weights
    output_layer_weights = np.array([
        np.mean(dJdW_ref[4]),
        np.mean(dJdW_ref[5]),
    ]).reshape(2, 1)

    # Compare weights
    np.testing.assert_allclose(hidden_layer_weights, dJdW_fc[0]), "Weight updates (hidden layer) do not match"
    np.testing.assert_allclose(output_layer_weights, dJdW_fc[1]), "Weight updates (output layer) do not match"

    # Aggregate hidden layer biases
    hidden_layer_biases = np.array([
        [np.mean(dJdB_ref[0]),    np.mean(dJdB_ref[1]),],
    ])

    # Aggregate output layer biases
    output_layer_biases = np.array([
        [np.mean(dJdB_ref[2])],
    ])

    # Compare biases
    np.testing.assert_allclose(hidden_layer_biases, dJdB_fc[0]), "Bias updates (hidden layer) do not match"
    np.testing.assert_allclose(output_layer_biases, dJdB_fc[1]), "Bias updates (output layer) do not match"
