"""
Functions to generate characteristic data to test the "shape" of a trading rule without wasting sample data.
"""
import numpy as np
import pandas as pd


def sample_from_distribution(distribution_type="normal", *args, **kwargs):
    """
    Generic sampling function to use

    Parameters
    ----------
    :distribution_type: str
        The name of the distribution we want to use.
    :*args:
        positional args to pass to sampling functions
    :**kwargs:
        keyword args to pass to sampling function
    """
    if distribution_type.lower() == "normal":
        return np.random.normal(*args, **kwargs)

    elif distribution_type.lower() == "uniform":
        return np.random.uniform(*args, **kwargs)

    # Add additional distributions as necessary

    else:
        raise NotImplementedError("Distribution type {0} not supported.".format(distribution_type))