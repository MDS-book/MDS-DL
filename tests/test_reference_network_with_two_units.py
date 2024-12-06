import pytest
import numpy as np
from tests.reference_network_with_two_units import ReferenceNetworkWith2Units

class TestReferenceNetworkHardCoded:
    def test_scalar_hardcoded(self):
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


class TestReferenceNetworkWith2UnitsVectorized:
    # We rely on the correctness established by the scalar hardcoded test above.
    # Now we test vectorized inputs.

    @pytest.mark.parametrize("W,B", [
        ((0.1, 0.2, -0.3, 0.4, 0.5, -0.6), (0.1, -0.2, -0.1)),
        ((-0.5, 0.3, 0.3, -0.4, -0.1, 0.2), (0.0, 0.1, 0.2)),
    ])
    def test_single_record(self, W, B):
        X1 = np.array([0.5])
        X2 = np.array([-1.0])
        Y = np.array([0.7])

        a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)
        Y_pred = ReferenceNetworkWith2Units.predict(W, B, X1, X2)
        np.testing.assert_allclose(Y_pred, a3, err_msg="predict does not match forward_pass output for single vectorized record")

        dJdW, dJdB = ReferenceNetworkWith2Units.backward_pass(W, B, X1, X2, Y)

        # Compare per-record to scalar version if it matches the scenario
        # If W,B,X1,X2,Y match the hardcoded scenario above, check against it:
        if W == (0.1, 0.2, -0.3, 0.4, 0.5, -0.6) and B == (0.1, -0.2, -0.1):
            # Expected from hardcoded test:
            a1_exp, a2_exp, y1_exp, y2_exp, a3_exp = -0.05, -0.75, 0.4875, 0.3206, -0.0486
            dJdW_exp = (-0.04665, 0.0933, 0.0489, -0.0978, -0.365, -0.24)
            dJdB_exp = (-0.0933, 0.0978, -0.74861)

            np.testing.assert_allclose(a1[0], a1_exp, rtol=1e-4, err_msg="a1 mismatch")
            np.testing.assert_allclose(a2[0], a2_exp, rtol=1e-4, err_msg="a2 mismatch")
            np.testing.assert_allclose(y1[0], y1_exp, rtol=1e-4, err_msg="y1 mismatch")
            np.testing.assert_allclose(y2[0], y2_exp, rtol=1e-4, err_msg="y2 mismatch")
            np.testing.assert_allclose(a3[0], a3_exp, rtol=1e-4, err_msg="a3 mismatch")

            for (v_grad, s_grad) in zip(dJdW, dJdW_exp):
                np.testing.assert_allclose(v_grad[0], s_grad, rtol=1e-3, err_msg="Weight gradient mismatch")
            for (v_grad, s_grad) in zip(dJdB, dJdB_exp):
                np.testing.assert_allclose(v_grad[0], s_grad, rtol=1e-3, err_msg="Bias gradient mismatch")

    @pytest.mark.parametrize("W,B", [
        ((0.1, 0.2, -0.3, 0.4, 0.5, -0.6), (0.1, -0.2, -0.1)),
        ((-0.5, 0.3, 0.3, -0.4, -0.1, 0.2), (0.0, 0.1, 0.2)),
    ])
    def test_three_records(self, W, B):
        X1 = np.array([0.5, -0.1, 0.2])
        X2 = np.array([-1.0, 0.3, 0.0])
        Y = np.array([0.7, 0.0, -0.5])

        a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)
        Y_pred = ReferenceNetworkWith2Units.predict(W, B, X1, X2)
        np.testing.assert_allclose(Y_pred, a3, err_msg="predict does not match forward_pass output for multiple records")

        dJdW, dJdB = ReferenceNetworkWith2Units.backward_pass(W, B, X1, X2, Y)

        # If we trust the scalar scenario for the first record (the hardcoded one),
        # we can compare the first record against that known scenario if it matches.
        if W == (0.1, 0.2, -0.3, 0.4, 0.5, -0.6) and B == (0.1, -0.2, -0.1):
            # For the first record (i=0), we can compare:
            i = 0
            a1_exp, a2_exp, y1_exp, y2_exp, a3_exp = -0.05, -0.75, 0.4875, 0.3206, -0.0486
            dJdW_exp = (-0.04665, 0.0933, 0.0489, -0.0978, -0.365, -0.24)
            dJdB_exp = (-0.0933, 0.0978, -0.74861)

            np.testing.assert_allclose(a1[i], a1_exp, rtol=1e-4, err_msg=f"a1 mismatch at record {i}")
            np.testing.assert_allclose(a2[i], a2_exp, rtol=1e-4, err_msg=f"a2 mismatch at record {i}")
            np.testing.assert_allclose(y1[i], y1_exp, rtol=1e-4, err_msg=f"y1 mismatch at record {i}")
            np.testing.assert_allclose(y2[i], y2_exp, rtol=1e-4, err_msg=f"y2 mismatch at record {i}")
            np.testing.assert_allclose(a3[i], a3_exp, rtol=1e-4, err_msg=f"a3 mismatch at record {i}")

            for (v_grad, s_grad) in zip(dJdW, dJdW_exp):
                np.testing.assert_allclose(v_grad[i], s_grad, rtol=1e-3, err_msg=f"Weight gradient mismatch at record {i}")
            for (v_grad, s_grad) in zip(dJdB, dJdB_exp):
                np.testing.assert_allclose(v_grad[i], s_grad, rtol=1e-3, err_msg=f"Bias gradient mismatch at record {i}")
