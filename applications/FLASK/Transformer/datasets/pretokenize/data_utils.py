import numpy as np


def random_zero_array(arr, p, mask_token):
    """
    Randomly zero out elements of an array with probability p
    """
    mask = np.random.choice([0, 1], size=arr.shape, p=[p, 1 - p])
    return arr * mask + mask_token * (1 - mask)
