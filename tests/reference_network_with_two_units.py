import numpy as np
from numpy import ndarray


class ReferenceNetworkWith2Units:
    """Reference Solution of a Network with 2 units and 1 output.

    The reference implementation is that of the network shown in Fig. 18.2 of the MDS-book.
    The implementation is that from Listing 18.2 of the MDS-book, included below:

    The class contains functions for computing the forwards and backward pass
    for a network of 2 neurons with sigmoid activation and 2 inputs,
    and a single output. As activation function, the sigmoidal function is hard-coded.
    Input (X1, X2) -> [FCL(2->2)] -> [Sigmoid AL] -> [FCL(2->1)] -> Output (Y_pred)

    The first unit has weights w1 and w2 (corresponding to input x1 and x2), the
    second unit has weights w3 and w4. Bias b1 is for unit 1, bias b2 is for unit2.
    w5 and w6 are the weights for the output/summation unit.
    """
    #
    @staticmethod
    def _sigmoidal_function(x: [float, ndarray]) -> [float, ndarray]:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoidal_derivative(x: [float, ndarray]) -> [float, ndarray]:
        s = ReferenceNetworkWith2Units._sigmoidal_function(x)
        return s * (1 - s)

    @staticmethod
    def forward_pass(W, B, X1, X2):
        (w1, w2, w3, w4, w5, w6), (b1, b2, b3) = W, B

        a1 = b1 + X1 * w1 + X2 * w2
        a2 = b2 + X1 * w3 + X2 * w4
        y1 = ReferenceNetworkWith2Units._sigmoidal_function(a1)
        y2 = ReferenceNetworkWith2Units._sigmoidal_function(a2)
        a3 = y1 * w5 + y2 * w6 + b3
        return a1, a2, y1, y2, a3

    @staticmethod
    def predict(W, B, X1, X2):
        _, _, _, _, Y_pred = ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)
        return Y_pred

    @staticmethod
    def backward_pass(W, B, X1, X2, Y):
        w1, w2, w3, w4, w5, w6 = W
        a1, a2, y1, y2, a3 = ReferenceNetworkWith2Units.forward_pass(W, B, X1, X2)

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
