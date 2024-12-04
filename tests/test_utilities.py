import pytest
import numpy as np
from mdsdl.utilities import \
    sigmoidal_function, \
    sigmoidal_derivative, \
    tanh_function, \
    tanh_derivative, \
    ReLU, \
    ReLU_derivative, \
    MSE, \
    MSE_derivative

def test_sigmoidal_function():
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    result = sigmoidal_function(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_sigmoidal_derivative():
    x = np.array([-1, 0, 1])
    expected = np.array([0.19661193, 0.25, 0.19661193])
    result = sigmoidal_derivative(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_tanh_function():
    x = np.array([-1, 0, 1])
    expected = np.array([-0.76159416, 0.0, 0.76159416])
    result = tanh_function(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_tanh_derivative():
    x = np.array([-1, 0, 1])
    expected = np.array([0.41997434, 1.0, 0.41997434])
    result = tanh_derivative(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_ReLU():
    x = np.array([-1, 0, 1])
    expected = np.array([0.0, 0.0, 1.0])
    result = ReLU(x)
    np.testing.assert_array_equal(result, expected)

def test_ReLU_derivative():
    x = np.array([-1, 0, 1])
    expected = np.array([0.0, 0.0, 1.0])
    result = ReLU_derivative(x)
    np.testing.assert_array_equal(result, expected)

def test_MSE():
    # Test case where predictions match the true values exactly
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    expected = 0.0
    result = MSE(y_true, y_pred)
    assert result == expected, f"Expected MSE of 0.0, got {result}"

    # Test with differing true values and predictions
    y_true = np.array([1, 2, 3])
    y_pred = np.array([4, 5, 6])
    # Manually compute the squared differences to validate the square operation
    # differences = y_true - y_pred  # [-3, -3, -3]
    # squared_differences = differences ** 2  # [9, 9, 9]
    expected = 9 #np.mean(squared_differences)  # (9 + 9 + 9) / 3 = 9.0
    result = MSE(y_true, y_pred)
    assert result == expected, f"Expected MSE of {expected}, got {result}"

    # Additional test with negative values to ensure squares are positive
    y_true = np.array([-1, -2, -3])
    y_pred = np.array([1, 2, 3])
    #differences = y_true - y_pred  # [-2, -4, -6]
    #squared_differences = differences ** 2  # [4, 16, 36]
    expected = 18.6667  # np.mean(squared_differences)  # (4 + 16 + 36) / 3 = 56 / 3 ≈ 18.6667
    result = MSE(y_true, y_pred)
    assert np.isclose(result, expected), f"Expected MSE of {expected}, got {result}"

def test_MSE_derivative():
    # Test derivative when predictions are greater than true values
    y_true = np.array([1, 2, 3])
    y_pred = np.array([4, 5, 6])
    expected = 2 / y_pred.size * (y_pred - y_true)  # [2, 2, 2]
    result = MSE_derivative(y_true, y_pred)
    np.testing.assert_array_equal(result, expected)

    # Test derivative with negative differences
    y_true = np.array([1, 2, 3])
    y_pred = np.array([-2, -1, 0])
    expected = 2 / y_pred.size * (y_pred - y_true)  # [-2, -2, -2]
    result = MSE_derivative(y_true, y_pred)
    np.testing.assert_array_equal(result, expected)

    # Test derivative when predictions equal true values
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    expected = np.zeros_like(y_true)
    result = MSE_derivative(y_true, y_pred)
    np.testing.assert_array_equal(result, expected)