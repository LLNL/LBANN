import lbann
import numpy as np
import test_util
import pytest


@pytest.mark.parametrize('constant', [0, 1])
@test_util.lbann_test()
def test_simple(constant):
    np.random.seed(20230515)
    x = np.random.rand(2, 2).astype(np.float32)
    ref = x + constant

    tester = test_util.ModelTester()
    x = tester.inputs(x)
    ref = tester.make_reference(ref)

    # Test layer
    y = lbann.AddConstant(x, constant=constant)
    tester.set_loss(lbann.MeanSquaredError(y, ref))
    return tester
