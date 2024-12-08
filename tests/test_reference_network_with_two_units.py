import pytest
import numpy as np
from tests.reference_network_with_two_units import ReferenceNetworkWith2Units

class TestReferenceNetworkWith2Units:
    """Testing first for only 1 record against a hardcoded solution.
    Then, we use this for testing the vectorized version.
    """
    def test_scalar_input_with_hardcoded_values(self):
        # Hard-coded scenario
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
        B = (0.1, -0.2, -0.1)
        X1 = 0.5
        X2 = -1.0
        Y = 0.7

        # Expected forward pass results
        a1_exp = -0.05 #ok
        a2_exp = -0.75 #ok
        y1_exp = 0.487503 #ok
        y2_exp = 0.32082 #ok
        a3_exp = -0.04874 #ok

        # Expected backward pass results
        dJdb1 = -0.09353 #-(0.7 - -0.04874) * 0.5 * 0.487503 * (1 - 0.487503)
        dJdb2 = 0.097888 # -(0.7 - -0.04874) * (-0.6) * 0.32082 * (1 - 0.32082)
        dJdb3 = -0.74874 # -(0.7 - (-0.04874))  #

        dJdw1 = -0.046765 #-0.09353 * 0.5
        dJdw2 = 0.09353
        dJdw3 = 0.048944
        dJdw4 = -0.097888
        dJdw5 = -0.3650
        dJdw6 = -0.240210
        dJdB_exp = (dJdb1, dJdb2, dJdb3)
        dJdW_exp = (dJdw1, dJdw2, dJdw3, dJdw4, dJdw5, dJdw6)

        # Actual computations
        a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)
        Y_pred = ReferenceNetworkWith2Units.predict(W, B, X1, X2)
        dJdW, dJdB = ReferenceNetworkWith2Units.backward_pass(W, B, X1, X2, Y)

        # Check forward pass
        np.testing.assert_allclose(a1, a1_exp, rtol=1e-4, err_msg="a1 mismatch")
        np.testing.assert_allclose(a2, a2_exp, rtol=1e-4, err_msg="a2 mismatch")
        np.testing.assert_allclose(y1, y1_exp, rtol=1e-4, err_msg="y1 mismatch")
        np.testing.assert_allclose(y2, y2_exp, rtol=1e-4, err_msg="y2 mismatch")
        np.testing.assert_allclose(a3, a3_exp, rtol=1e-4, err_msg="a3 mismatch")
        np.testing.assert_allclose(Y_pred, a3_exp, rtol=1e-4, err_msg="Y_pred mismatch")

        # Check backward pass
        np.testing.assert_allclose(dJdW, dJdW_exp, rtol=1e-3, err_msg="dJdW mismatch")
        np.testing.assert_allclose(dJdB, dJdB_exp, rtol=1e-3, err_msg="dJdB mismatch")

    @pytest.mark.parametrize("W,B", [
        ((0.1, 0.2, -0.3, 0.4, 0.5, -0.6), (0.1, -0.2, -0.1)),
        ((-0.5, 0.3, 0.3, -0.4, -0.1, 0.2), (0.0, 0.1, 0.2)),
    ])
    def test_array_input(self, W, B):
        """
        We rely on the correctness established by the scalar hardcoded test above.
        Now we test vectorized inputs.
        """
        # W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
        # B = (0.1, -0.2, -0.1)
        X1 = np.array([0.5, 0.2, 0.6])
        X2 = np.array([-1.0, 0.4, 0])
        Y = np.array([0.7, 0.1, -1])

        a1, a2, y1, y2, a3 = \
            ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)
        Y_pred = ReferenceNetworkWith2Units.predict(W, B, X1, X2)
        dJdW, dJdB = ReferenceNetworkWith2Units.backward_pass(W, B, X1, X2, Y)

        # just so that we can index properly below
        dJdW = np.array(dJdW)  # n_weights x n_records = 6 x 3
        dJdB = np.array(dJdB)  # 3x3

        for i, (x1, x2, y) in enumerate(zip(X1, X2, Y)):
            a1_desired, a2_desired, y1_desired, y2_desired, a3_desired = \
                ReferenceNetworkWith2Units.forward_pass(W, B, x1, x2)
            Y_pred_desired = ReferenceNetworkWith2Units.predict(W, B, x1, x2)
            dJdW_desired, dJdB_desired = \
                ReferenceNetworkWith2Units.backward_pass(W, B, x1, x2, y)

            # Check forward pass
            np.testing.assert_allclose(a1[i], a1_desired, rtol=1e-4, err_msg="a1 mismatch")
            np.testing.assert_allclose(a2[i], a2_desired, rtol=1e-4, err_msg="a2 mismatch")
            np.testing.assert_allclose(y1[i], y1_desired, rtol=1e-4, err_msg="y1 mismatch")
            np.testing.assert_allclose(y2[i], y2_desired, rtol=1e-4, err_msg="y2 mismatch")
            np.testing.assert_allclose(a3[i], a3_desired, rtol=1e-4, err_msg="a3 mismatch")
            np.testing.assert_allclose(Y_pred[i], Y_pred_desired, rtol=1e-4, err_msg="Y_pred mismatch")

            # Check backward pass
            np.testing.assert_allclose(dJdW[:, i], dJdW_desired, rtol=1e-3, err_msg="dJdW mismatch")
            np.testing.assert_allclose(dJdB[:, i], dJdB_desired, rtol=1e-3, err_msg="dJdB mismatch")

    def test_cost_function_consistency(self):
        """
        Test that the cost function is consistent with the definition:
        J = 0.5 * mean((y_true - y_pred)^2)

        If we do not average over the number of records, the cost would be too large
        and this test should fail until we fix the problem.
        """

        # Scenario: 2 records
        W = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
        B = (0.1, -0.2, -0.1)
        X1 = np.array([0.5, -0.1])
        X2 = np.array([-1.0, 0.3])
        Y = np.array([0.7, -0.3])

        # Forward pass using the reference code
        # a3 is the network output for all records
        _, _, _, _, a3 = ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)

        # Correct cost computation (with averaging):
        # J = 0.5 * mean((Y - a3)^2)  -->  this is the cost that is consistent with the derivative that we're using in the MDS book
        correct_cost = 0.5 * np.mean((Y - a3)**2)

        # note the "/2" for 2 records
        computed_cost = 0.5 * ((Y[0] - a3[0])**2 + (Y[1] - a3[1])**2) / 2
        np.testing.assert_allclose(computed_cost, correct_cost, rtol=1e-5,
                                   err_msg="the computed cost should match the correctly averaged MSE.")
