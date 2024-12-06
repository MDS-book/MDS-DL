import pytest
import numpy as np
from mdsdl.utilities import tanh_function, tanh_derivative, sigmoidal_function, sigmoidal_derivative
from mdsdl.fully_connected import FullyConnectedLayer, ActivationLayer






class TestFullyConnectedLayer:
    def test_fully_connected_layer_initialization(self):
        n_inputs = 3
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs, seed=42)

        assert layer.weights.shape == (n_inputs, n_outputs), "Weights shape mismatch"
        assert layer.biases.shape == (1, n_outputs), "Biases shape mismatch"
        assert np.all(layer.weights >= -0.5) and np.all(layer.weights <= 0.5), "Weights not in expected range"
        assert np.all(layer.biases >= -0.5) and np.all(layer.biases <= 0.5), "Biases not in expected range"

    def test_fully_connected_layer_feed_forward(self):
        n_inputs = 2
        n_outputs = 1
        layer = FullyConnectedLayer(n_inputs, n_outputs)
        # Set weights and biases to known values
        layer.weights = np.array([[0.5], [0.5]])
        layer.biases = np.array([[0.1]])
        x = np.array([[1, 2]])
        expected_output = np.dot(x, layer.weights) + layer.biases  # [[1*0.5 + 2*0.5 + 0.1]] = [[1.6]]
        print(expected_output)
        output = layer.feed_forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, err_msg="Feed forward output mismatch")

    def test_fully_connected_layer_feed_forward_2(self):
        w1, w2, b1 = 0.1, 0.4, 0.5  # first unit
        w3, w4, b2 = -0.7, 0.3, 0.1  # second unit
        w5, w6, b3 = 0.5, -0.3, 0.1  # output node
        W = (w1, w2, w3, w4, w5, w6)
        B = (b1, b2, b3)
        X = np.array([
            [1.1, -2.5],
            [5.0, 1.0],
        ])
        X1 = X[:, 0]
        X2 = X[:, 1]

        y1 = b1 + w1 * X1 + w2 * X2
        y2 = b2 + w3 * X1 + w4 * X2
        #print(np.stack([y1, y2], axis=1))

        # A hidden layer with two inputs per unit (n_inputs=2) and
        # two units, each of which has 1 output (n_outputs=2)
        layer = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        layer.weights = np.array([
            [w1, w3],
            [w2, w4]
        ])
        layer.biases = np.array([[b1, b2]])
        y = layer.feed_forward(X)

        np.testing.assert_allclose(actual=y,
                                   desired=np.stack([y1, y2], axis=1),
                                   err_msg="forward pass difference")


    # def test_fully_connected_layer_feed_forward_with_sigmoid(self):
    #     w1, w2, b1 =  0.1, 0.4, 0.5  # first unit
    #     w3, w4, b2 = -0.7, 0.3, 0.1  # second unit
    #     w5, w6, b3 = 0.5, -0.3, 0.1  # output node
    #     W = (w1, w2, w3, w4, w5, w6)
    #     B = (b1, b2, b3)
    #     X = np.array([
    #         [1.1, -2.5],
    #         [5.0, 1.0],
    #     ])
    #     X1 = X[:, 0]
    #     X2 = X[:, 1]
    #
    #     a1, a2, y1, y2, a3 = forward_pass(W, B, X1, X2)
    #     print("\nana. output:", a1, a2)
    #     print("ana. output:", y1, y2)
    #     print(np.stack([y1, y2], axis=1))
    #
    #     # A hidden layer with two inputs per unit (n_inputs=2) and
    #     # two units, each of which has 1 output (n_outputs=2)
    #     layer = FullyConnectedLayer(n_inputs=2, n_outputs=2)
    #     layer.weights = np.array([
    #         [w1, w3],
    #         [w2, w4]
    #     ])
    #     layer.biases = np.array([[b1, b2]])
    #     y = layer.feed_forward(X)
    #     print("\nFCN output:", y)
    #     print("x:", layer.x, "\nw:", layer.weights, "\nb:", layer.biases)
    #
    #     np.testing.assert_allclose(actual=y,
    #                                desired=np.stack([y1, y2], axis=1),
    #                                err_msg="forward pass difference")


    def test_fully_connected_layer_backward_propagation(self):
        n_inputs = 2
        n_outputs = 1
        layer = FullyConnectedLayer(n_inputs, n_outputs)

        # given parameter and training data
        W = np.array([[0.5],
                      [0.5]])
        b = np.array([[0.1]])
        learning_rate = 0.1
        x = np.array([[1, 2]])

        dJdy = np.array([[1]])  # we simply assume that this is given!


        # manual calculation
        y_predicted = b[0, 0] + x[0, 0] * W[0, 0]  + x[0, 1] * W[1, 0]
        dJdW = np.dot(x.T, dJdy)
        dJdb = dJdy
        dJdy_prev = np.dot(dJdy, W.T)  # this is the same as dJdx!
        dJdx_desired = dJdy_prev

        W_new = W - learning_rate * dJdW
        b_new = b - learning_rate * dJdb

        # using the FCN class
        layer.weights = W
        layer.biases = b
        layer.feed_forward(x)
        dJdx, dJdW, dJdb = layer.backward_propagation(dJdy, learning_rate)

        # Check weights and biases update
        np.testing.assert_array_almost_equal(layer.weights, W_new,
                                             err_msg="Weights not updated correctly")
        np.testing.assert_array_almost_equal(layer.biases, b_new,
                                             err_msg="Biases not updated correctly")
        np.testing.assert_array_almost_equal(dJdx, dJdx_desired,
                                             err_msg="dJdx (=dJdy_prev) not computed correctly")

        return


    def test_fully_connected_layer_feed_forward_multiple_records(self):
        n_inputs = 2
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs)
        # Set weights and biases to known values
        # the weights of the first unit are in the first column
        layer.weights = np.array([[0.1, 0.2],
                                  [0.3, 0.4]])
        layer.biases = np.array([[0.5, 0.6]])
        # Input with multiple records (batch size = 3)
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        # Manually compute expected outputs
        # For each record: output = x * weights + biases
        #expected_output = np.dot(x, layer.weights) + layer.biases
        expected_output = np.array(
         [[1*0.1 + 2*0.3 + 0.5, 1*0.2 + 2*0.4 + 0.6],
          [3*0.1 + 4*0.3 + 0.5, 3*0.2 + 4*0.4 + 0.6],
          [5*0.1 + 6*0.3 + 0.5, 5*0.2 + 6*0.4 + 0.6]])
        output = layer.feed_forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, err_msg="Feed forward output mismatch with multiple records")

    def test_fully_connected_layer_backward_propagation_multiple_records(self):
        n_inputs = 2
        n_outputs = 2
        layer = FullyConnectedLayer(n_inputs, n_outputs)
        # Set weights and biases to known values
        layer.weights = np.array([[0.1, 0.2],
                                  [0.3, 0.4]])
        layer.biases = np.array([[0.5, 0.6]])
        # Input with multiple records
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        layer.feed_forward(x)

        # Make up a gradient of loss with respect to output (dJ/dy)
        dJdy = np.array([[0.1, 0.2],
                         [0.3, 0.4],
                         [0.5, 0.6]])
        learning_rate = 0.01

        expected_dJdx = np.dot(dJdy, layer.weights.T)  # [[0.5, 0.5]]


        # # Manually compute gradients dJdW = x.T dot dJdy
        # expected_dJdW = np.dot(x.T, dJdy)
        #
        # # dJdb = sum over batch of dJdy
        # expected_dJdb = np.sum(dJdy, axis=0, keepdims=True)
        #
        # # Store old weights and biases for comparison after update
        # old_weights = layer.weights.copy()
        # old_biases = layer.biases.copy()

        # Perform backward propagation
        dJdx, dJdW, dJdb = layer.backward_propagation(dJdy, learning_rate)
        np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Backward propagation output mismatch")
        #
        # # Check weights and biases update
        # expected_weights = old_weights - learning_rate * expected_dJdW
        # expected_biases = old_biases - learning_rate * expected_dJdb
        # np.testing.assert_array_almost_equal(layer.weights, expected_weights, err_msg="Weights not updated correctly with multiple records")
        # np.testing.assert_array_almost_equal(layer.biases, expected_biases, err_msg="Biases not updated correctly with multiple records")
        # # Check dJdx (gradient w.r.t input)
        # expected_dJdx = np.dot(dJdy, old_weights.T)
        # np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Backward propagation output mismatch with multiple records")

class TestActivationLayer:
    def test_activation_layer_feed_forward(self):
        activation_layer = ActivationLayer(activation_function=tanh_function,
                                           activation_derivative=tanh_derivative)
        x = np.array([[0, 1]])
        expected_output = np.tanh(x)
        output = activation_layer.feed_forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, err_msg="Activation feed forward output mismatch")

    def test_activation_layer_backward_propagation(self):
        activation_layer = ActivationLayer(activation_function=tanh_function,
                                           activation_derivative=tanh_derivative)
        x = np.array([[0, 1]])
        activation_layer.feed_forward(x)
        dJdy = np.array([[1, 1]])
        expected_dJdx = (1 - np.tanh(x) ** 2) * dJdy
        dJdx, dJdW, dJdb = activation_layer.backward_propagation(dJdy, learning_rate=None)
        np.testing.assert_array_almost_equal(dJdx, expected_dJdx, err_msg="Activation backward propagation output mismatch")


class TestWithReferenceNetwork:

    ######################################################################################################
    # These are some helper functions for computing the forwards and backward pass
    # for a network of 2 neurons, 2 inputs and a single output. As activation
    # function, the sigmoidal function is hard-coded.
    # Input (X1, X2) -> [FCL(2->2)] -> [Sigmoid AL] -> [FCL(2->1)] -> Output (Y_pred)
    #
    def _sigmoidal_function(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoidal_derivative(self, x):
        s = sigmoidal_function(x)
        return s * (1 - s)

    def _forward_pass(self, W, B, X1, X2):
        (w1, w2, w3, w4, w5, w6), (b1, b2, b3) = W, B

        a1 = b1 + X1 * w1 + X2 * w2
        a2 = b2 + X1 * w3 + X2 * w4
        y1 = self._sigmoidal_function(a1)
        y2 = self._sigmoidal_function(a2)
        a3 = y1 * w5 + y2 * w6 + b3
        return a1, a2, y1, y2, a3

    def _predict(self, W, B, X1, X2):
        _, _, _, _, Y_pred = self._forward_pass(W, B, X1, X2)
        return Y_pred

    def _backward_pass(self, W, B, X1, X2, Y):
        w1, w2, w3, w4, w5, w6 = W
        a1, a2, y1, y2, a3 = self._forward_pass(W, B, X1, X2)

        dJdb1 = -(Y - a3) * w5 * y1 * (1 - y1)
        dJdw1 = dJdb1 * X1
        dJdw2 = dJdb1 * X2
        dJdb2 = -(Y - a3) * w6 * y2 * (1 - y2)
        dJdw3 = dJdb2 * X1
        dJdw4 = dJdb2 * X2
        dJdb3 = -(Y - a3)
        dJdw5 = -(Y - a3) * y1
        dJdw6 = -(Y - a3) * y2

        dJdW, dJdB = (dJdw1, dJdw2, dJdw3, dJdw4, dJdw5, dJdw6), (dJdb1, dJdb2, dJdb3)
        return dJdW, dJdB
    ######################################################################################################

    def test_two_unit_network_against_reference_with_1_record(self):
        # Given parameters (example values)
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)  # (w1,w2,w3,w4,w5,w6)
        B = (0.1, -0.2, -0.1)  # (b1,b2,b3)
        X1, X2 = 0.5, -1.0
        Y = 0.7

        # Compute reference values
        Y_pred_ref = self._predict(W, B, X1, X2)
        dJdW_ref, dJdB_ref = self._backward_pass(W, B, X1, X2, Y)

        # Set up the class-based network
        fcl1 = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        # Assign weights and biases to match reference
        fcl1.weights = np.array([[W[0], W[2]],
                                 [W[1], W[3]]])
        fcl1.biases = np.array([[B[0], B[1]]])

        al = ActivationLayer(activation_function=sigmoidal_function,
                             activation_derivative=sigmoidal_derivative)

        fcl2 = FullyConnectedLayer(n_inputs=2, n_outputs=1)
        fcl2.weights = np.array([[W[4]],
                                 [W[5]]])
        fcl2.biases = np.array([[B[2]]])

        a1, a2, y1, y2, y3 = self._forward_pass(W, B, X1, X2)
        print("\nana. a1:", a1, "  a2:", a2)
        print("     y1:", y1, "  y2:", y2, "  y3:", y3)
        # a1 = 0.1 + 0.1 * 0.5 + 0.2 * (-1) = -0.05

        # Forward pass with class-based layers
        x_input = np.array([[X1, X2]])
        y_pred = fcl1.feed_forward(x_input)
        print("a1:", y_pred)

        y_pred = al.feed_forward(y_pred)
        print("y1:", y_pred)


        y_pred = fcl2.feed_forward(y_pred)
        print("y3:", y_pred)


        Y_pred_class = y_pred[0, 0]

        # Check forward pass consistency
        np.testing.assert_almost_equal(Y_pred_class, Y_pred_ref, decimal=7, err_msg="Forward pass mismatch")
        #---------------------------------------------

        # Now for backward pass:
        # We'll compute gradients in the class-based approach. We must define a cost derivative:
        # dJ/dy for MSE = (a3 - Y)
        dJdy = np.array([[Y_pred_class - Y]])  # shape (1, 1)

        # Backprop through fcl2
        dJdy_prev, fcl2_last_dJdW, fcl2_dJdb = fcl2.backward_propagation(dJdy, learning_rate=0.0)

        # Backprop through activation layer
        dJdy_prev, _, _ = al.backward_propagation(dJdy_prev, learning_rate=0.0)

        # Backprop through fcl1
        dJdy_prev, fcl1_dJdW, fcl1_dJdb = fcl1.backward_propagation(dJdy_prev, learning_rate=0.0)

        # At this point, we need the gradients. Let's assume we've modified our FullyConnectedLayer
        # to store the last computed gradients in attributes `last_dJdW` and `last_dJdb` for testing.
        # If not, you would need to adapt your code accordingly.

        # Extract gradients from class-based layers
        # fcl1: weights shape (2,2) corresponds to w1,w2 in row 0, w3,w4 in row 1
        # fcl2: weights shape (2,1) corresponds to w5,w6
        print(fcl1_dJdW.shape)
        dJdw1_class, dJdw2_class = fcl1_dJdW[0, 0], fcl1_dJdW[1, 0]
        dJdw3_class, dJdw4_class = fcl1_dJdW[0, 1], fcl1_dJdW[1, 1]
        dJdw5_class, dJdw6_class = fcl2_last_dJdW[0, 0], fcl2_last_dJdW[1, 0]

        dJdw1_class, dJdw2_class = fcl1_dJdW[0, 0], fcl1_dJdW[1, 0]
        dJdw3_class, dJdw4_class = fcl1_dJdW[0, 1], fcl1_dJdW[1, 1]

        dJdW_class = (dJdw1_class, dJdw2_class, dJdw3_class, dJdw4_class, dJdw5_class, dJdw6_class)
        dJdB_class = (dJdb1_class, dJdb2_class, dJdb3_class)

        # Compare gradients
        np.testing.assert_almost_equal(dJdW_class, dJdW_ref, decimal=7, err_msg="Weight gradients mismatch")
        np.testing.assert_almost_equal(dJdB_class, dJdB_ref, decimal=7, err_msg="Bias gradients mismatch")
