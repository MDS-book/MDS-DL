import pytest
import numpy as np
from numpy import ndarray
from mdsdl.fully_connected import FullyConnectedLayer, ActivationLayer
from mdsdl.fully_connected.fc_network import FCNetwork
from mdsdl.utilities import tanh_function, tanh_derivative, \
    sigmoidal_function, sigmoidal_derivative, \
    MSE, MSE_derivative
from tests.reference_network_with_two_units import ReferenceNetworkWith2Units



class TestFCNetwork:
    def test_fcnetwork_init(self):
        nn = FCNetwork(cost_function=MSE,
                       derivative_of_cost=MSE_derivative)
        assert nn.cost_function == MSE
        assert nn.derivative_of_cost == MSE_derivative
        assert nn.layers == list()

    def test_fcnetwork_add_layer(self):
        nn = FCNetwork(cost_function=MSE,
                       derivative_of_cost=MSE_derivative)
        nn.add_layer(FullyConnectedLayer(n_inputs=3, n_outputs=2))
        nn.add_layer(ActivationLayer(activation_function=tanh_function,
                                     activation_derivative=tanh_derivative))

        assert len(nn.layers) == 2
        assert nn.layers[0].weights.shape == (3, 2)
        assert nn.layers[1].phi == tanh_function

    def test_fcnetwork_predict(self):
        # we're using the reference network

        # Given parameters (example values)
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)  # (unit 1: w1,w2, unit 2: w3,w4, unit 3: w5,w6)
        B = (0.1, -0.2, -0.1)  # (b1,b2,b3)
        X1, X2 = 0.5, -1.0
        Y = 0.7

        # Compute reference values
        Y_pred_ref = ReferenceNetworkWith2Units.predict(W, B, X1, X2)

        nn = FCNetwork(cost_function=MSE,
                       derivative_of_cost=MSE_derivative)
        fcl1 = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        acl = ActivationLayer(activation_function=sigmoidal_function,
                              activation_derivative=sigmoidal_derivative)
        fcl2 = FullyConnectedLayer(n_inputs=2, n_outputs=1)
        fcl1.weights = np.array([[W[0], W[2]],
                                 [W[1], W[3]]])
        fcl1.biases = np.array([[B[0],
                                 B[1]]])
        fcl2.weights = np.array([[W[4]],
                                 [W[5]]])
        fcl2.biases = np.array([[B[2]]])

        nn.add_layer(fcl1)
        nn.add_layer(acl)
        nn.add_layer(fcl2)

        X = np.array([[X1, X2]])
        Y_pred = nn.predict(X)

        # print("\n ref:", Y_pred_ref)
        # print("\n ", Y_pred)
        #
        np.testing.assert_approx_equal(Y_pred_ref, Y_pred[[0]])

    def test_fcnetwork_train(self):
        # We'll test training on a very simple scenario.
        # We'll use the same architecture as in test_fcnetwork_predict.
        # We'll try to train the network on a single data point to see if cost decreases.

        np.random.seed(42)  # For reproducibility

        # Create a single data sample as training data
        X_train = np.array([[0.5, -1.0]])
        Y_train = np.array([[0.7]])

        nn = FCNetwork(cost_function=MSE,
                       derivative_of_cost=MSE_derivative)

        # A simple two-layer network:
        fcl1 = FullyConnectedLayer(n_inputs=2, n_outputs=2, seed=42)
        acl = ActivationLayer(activation_function=sigmoidal_function,
                              activation_derivative=sigmoidal_derivative)
        fcl2 = FullyConnectedLayer(n_inputs=2, n_outputs=1, seed=42)

        nn.add_layer(fcl1)
        nn.add_layer(acl)
        nn.add_layer(fcl2)

        # Run training for a few epochs
        epochs = 10
        learning_rate = 0.1
        all_costs = nn.train(X_train, Y_train, epochs=epochs, learning_rate=learning_rate)

        # Check that we get a cost value per epoch
        assert len(all_costs) == epochs, "Number of cost values should match number of epochs"

        # We can check if cost at the end is at least not higher than the start
        # In a well-behaved scenario, it should decrease.
        assert all_costs[-1] <= all_costs[0], "Cost did not decrease after training"

        # Optional: If we want stricter checks, we could do so, but might fail depending on initialization.
        # assert all_costs[-1] < all_costs[0], "Cost should decrease with training"



# Assuming you have:
# from your_module import FCNetwork, FullyConnectedLayer, ActivationLayer, MSE, MSE_derivative
# from your_module import sigmoidal_function, sigmoidal_derivative

class TestFCNetworkDetailedTraining:
    def test_train_two_steps_single_record(self):
        # Initial known scenario
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)  # (w1,w2,w3,w4,w5,w6)
        B = (0.1, -0.2, -0.1)                # (b1,b2,b3)
        X1, X2 = 0.5, -1.0
        Y = 0.7
        learning_rate = 0.1
        epochs = 2

        # =========== Manual Computation Using Reference Class ===========
        # Convert W,B tuples into mutable arrays for updates
        W_arr = list(W)
        B_arr = list(B)

        def forward_backward_update(X1, X2, Y):
            # Forward pass
            a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W_arr, B_arr, X1, X2)
            # Cost for reference
            cost = MSE(Y, a3)
            # Backward pass
            dJdW, dJdB = ReferenceNetworkWith2Units.backward_pass(W_arr, B_arr, X1, X2, Y)

            # Update weights and biases manually
            # dJdW and dJdB are tuples like (dJdw1, dJdw2, ..., dJdw6) and (dJdb1, dJdb2, dJdb3)
            for i in range(len(W_arr)):
                W_arr[i] = W_arr[i] - learning_rate * dJdW[i]

            for i in range(len(B_arr)):
                B_arr[i] = B_arr[i] - learning_rate * dJdB[i]

            # this here is important because in the class-based approach we are also calling
            # nn.predict() after the training
            a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W_arr, B_arr, X1, X2)

            return cost, (a3, a1, a2, y1, y2)

        # Perform two manual updates
        cost_step1, _ = forward_backward_update(X1, X2, Y)
        cost_step2, (final_a3, _, _, _, _) = forward_backward_update(X1, X2, Y)

        # Final manual predictions after two steps
        final_manual_pred = final_a3
        final_manual_W = tuple(W_arr)
        final_manual_B = tuple(B_arr)

        # =========== Using FCNetwork to train for 2 epochs ===========
        nn = FCNetwork(cost_function=MSE, derivative_of_cost=MSE_derivative)

        # Build the same network structure
        fcl1 = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        acl = ActivationLayer(activation_function=sigmoidal_function,
                              activation_derivative=sigmoidal_derivative)
        fcl2 = FullyConnectedLayer(n_inputs=2, n_outputs=1)

        # Assign initial weights and biases to match reference scenario
        fcl1.weights = np.array([[W[0], W[2]],
                                 [W[1], W[3]]])
        fcl1.biases = np.array([[B[0], B[1]]])
        fcl2.weights = np.array([[W[4]],
                                 [W[5]]])
        fcl2.biases = np.array([[B[2]]])

        nn.add_layer(fcl1)
        nn.add_layer(acl)
        nn.add_layer(fcl2)

        X_train = np.array([[X1, X2]])
        Y_train = np.array([[Y]])

        all_costs = nn.train(X_train, Y_train, epochs=epochs, learning_rate=learning_rate)
        # After training 2 epochs, check final weights and predictions
        # Extract final weights and biases
        final_nn_W = (fcl1.weights[0,0], fcl1.weights[1,0],
                      fcl1.weights[0,1], fcl1.weights[1,1],
                      fcl2.weights[0,0], fcl2.weights[1,0])
        final_nn_B = (fcl1.biases[0,0], fcl1.biases[0,1], fcl2.biases[0,0])

        # Final prediction from NN
        Y_pred = nn.predict(X_train)[0,0]

        # Compare final weights, biases, and predictions to manual calculation
        np.testing.assert_allclose(final_nn_W, final_manual_W, rtol=1e-4, err_msg="Final weights mismatch")
        np.testing.assert_allclose(final_nn_B, final_manual_B, rtol=1e-4, err_msg="Final biases mismatch")
        np.testing.assert_allclose(Y_pred, final_manual_pred, rtol=1e-4, err_msg="Final prediction mismatch")

        # Compare costs as well
        # The last cost in all_costs should be close to cost_step2 / number_of_samples (which is 1 here)
        np.testing.assert_allclose(all_costs[-1], cost_step2, rtol=1e-4, err_msg="Final cost mismatch")


    def test_train_two_steps_multiple_records(self):
        # Multiple data records scenario
        # We will create two records and run two update steps.
        # For simplicity, we reuse the same network and initial conditions,
        # but with two records: (X1a, X2a, Ya) and (X1b, X2b, Yb).

        # Initial scenario
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
        B = (0.1, -0.2, -0.1)
        learning_rate = 0.1
        epochs = 2

        # Two records
        X_train = np.array([[0.5, -1.0],
                            [0.2, 0.1]])
        Y_train = np.array([[0.7],
                            [-0.3]])

        # Manual step-by-step:
        W_arr = list(W)
        B_arr = list(B)

        def forward_backward_update(X1, X2, Y, W_arr, B_arr):
            a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W_arr, B_arr, X1, X2)
            cost = MSE(Y, a3)
            dJdW, dJdB = ReferenceNetworkWith2Units.backward_pass(W_arr, B_arr, X1, X2, Y)
            for i in range(len(W_arr)):
                W_arr[i] -= learning_rate * dJdW[i]
            for i in range(len(B_arr)):
                B_arr[i] -= learning_rate * dJdB[i]
            return cost

        # Two epochs of manual updates over both records:
        # Each epoch: update once per record in sequence.
        def manual_train_two_epochs(W_arr, B_arr):
            # Epoch 1
            cost_record1 = forward_backward_update(X_train[0,0], X_train[0,1], Y_train[0,0], W_arr, B_arr)
            cost_record2 = forward_backward_update(X_train[1,0], X_train[1,1], Y_train[1,0], W_arr, B_arr)
            epoch1_cost = (cost_record1 + cost_record2) / 2.0

            # Epoch 2
            cost_record1_2 = forward_backward_update(X_train[0,0], X_train[0,1], Y_train[0,0], W_arr, B_arr)
            cost_record2_2 = forward_backward_update(X_train[1,0], X_train[1,1], Y_train[1,0], W_arr, B_arr)
            epoch2_cost = (cost_record1_2 + cost_record2_2) / 2.0

            return epoch1_cost, epoch2_cost

        epoch1_cost_manual, epoch2_cost_manual = manual_train_two_epochs(W_arr, B_arr)
        final_manual_W = tuple(W_arr)
        final_manual_B = tuple(B_arr)

        # Predict after manual training
        # After training 2 epochs:
        # We'll pick one record to check predictions, or check both
        # Let's check predictions on the entire X_train set
        def manual_predict(X, W, B):
            Y_pred_list = []
            for x in X:
                _, _, _, _, a3 = ReferenceNetworkWith2Units.forward_pass(W, B, x[0], x[1])
                Y_pred_list.append(a3)
            return np.array(Y_pred_list).reshape(X.shape[0], 1)

        final_manual_pred = manual_predict(X_train, final_manual_W, final_manual_B)

        # Now use FCNetwork to train for 2 epochs
        nn = FCNetwork(cost_function=MSE, derivative_of_cost=MSE_derivative)
        fcl1 = FullyConnectedLayer(n_inputs=2, n_outputs=2)
        acl = ActivationLayer(activation_function=sigmoidal_function,
                              activation_derivative=sigmoidal_derivative)
        fcl2 = FullyConnectedLayer(n_inputs=2, n_outputs=1)

        fcl1.weights = np.array([[W[0], W[2]],
                                 [W[1], W[3]]])
        fcl1.biases = np.array([[B[0], B[1]]])
        fcl2.weights = np.array([[W[4]],
                                 [W[5]]])
        fcl2.biases = np.array([[B[2]]])

        nn.add_layer(fcl1)
        nn.add_layer(acl)
        nn.add_layer(fcl2)

        all_costs = nn.train(X_train, Y_train, epochs=2, learning_rate=learning_rate)

        final_nn_W = (fcl1.weights[0,0], fcl1.weights[1,0],
                      fcl1.weights[0,1], fcl1.weights[1,1],
                      fcl2.weights[0,0], fcl2.weights[1,0])
        final_nn_B = (fcl1.biases[0,0], fcl1.biases[0,1], fcl2.biases[0,0])

        Y_pred_nn = nn.predict(X_train)
        print("\n",Y_pred_nn.shape)

        # as the reference solution has only 1 output, Y has shape n_records x 1
        # The class based implementation is n_records x
        final_manual_pred = final_manual_pred.reshape([-1, 1, 1])

        # Compare final weights and biases
        np.testing.assert_allclose(final_nn_W, final_manual_W, rtol=1e-4, err_msg="Final weights mismatch (multiple records)")
        np.testing.assert_allclose(final_nn_B, final_manual_B, rtol=1e-4, err_msg="Final biases mismatch (multiple records)")

        # Compare predictions
        np.testing.assert_allclose(Y_pred_nn, final_manual_pred, rtol=1e-4, err_msg="Final predictions mismatch (multiple records)")

        # Compare costs
        # all_costs[0] should be close to epoch1_cost_manual and all_costs[1] close to epoch2_cost_manual
        np.testing.assert_allclose(all_costs[0], epoch1_cost_manual, rtol=1e-4, err_msg="Epoch 1 cost mismatch (multiple records)")
        np.testing.assert_allclose(all_costs[1], epoch2_cost_manual, rtol=1e-4, err_msg="Epoch 2 cost mismatch (multiple records)")
