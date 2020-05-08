"""
Specify the weights for the top level of the hierarchy.
"""
# TODO: Add test that these sum to 1

STRATEGY_WEIGHTS = {
    'bias': .6,
    'trend': .4
}

INSTRUMENT_WEIGHTS = {
    'equity':  0.75,
    'bond': 0.125,
    'real_estate': 0.125
}