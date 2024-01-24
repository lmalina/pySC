import pytest
import numpy as np
from pySC.utils.sc_tools import pinv


def test_pinv(matrix):
    a = pinv(matrix, num_removed=0, alpha=0, damping=1, plot=False)
    a2 = pinv(matrix, num_removed=0, alpha=0, damping=0.9, plot=True)
    b = np.linalg.pinv(matrix)
    assert np.allclose(a, b)
    assert np.allclose(a2, b * 0.9)
    a3 = pinv(matrix, num_removed=0, alpha=0.1, damping=0.9, plot=True)


@pytest.fixture
def matrix():
    return np.array([[0.04, 0.9,  0.81, 0.96, 0.53],
                     [0.02, 0.07, 0.82, 0.56, 0.31],
                     [0.96, 0.75, 0.26, 0.82, 0.78],
                     [0.8,  0.17, 0.62, 0.44, 0.53],
                     [0.97, 0.97, 0.25, 0.86, 0.54]])
