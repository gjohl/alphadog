from numpy import cumsum, log
from numpy.random import randn


def test_hurst_exponent():
    # TODO TEST
    # Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
    gbm = log(cumsum(randn(100000))+1000)
    mr = log(randn(100000)+1000)
    tr = log(cumsum(randn(100000)+1)+1000)
