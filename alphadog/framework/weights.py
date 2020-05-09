"""
Functionality for setting/calculating the relative weights of different
strategies (fweights) and instruments (pweights).

# TODO:
As a first pass we hard code these weights and trade every strategy for every instrument.
This essentially follows the hand-crafted method in Systematic Trading, Carver.

- Add functionality to bootstrap data and optimise these weights, accounting for
  correlations between strategies/instruments.
- Account for missing items. If one strategy is missing, scale up all of the other weights
  so they still sum to 1
"""

STRATEGY_WEIGHTS = {
    'bias': {'weight': 0.6},
    'trend': {
        'weight': 0.4,
        'VMOM': {'weight': 0.5},
        'VBO': {'weight': 0.5}
    }
}

INSTRUMENT_WEIGHTS = {
    'equity': {
        'weight': 0.7,
        'uk': {'weight': 0.3},
        'europe': {'weight': 0.3},
        'us': {'weight': 0.4}
    },

    'bond': {
        'weight': 0.2,
        'government': {
            'weight': 0.3,
            'uk': {'weight': 0.3},
            'europe': {'weight': 0.3},
            'us': {'weight': 0.4}
        },
        'corporate': {
            'weight': 0.3,
            'global': {'weight': 1.},
        },
        'inflation': {
            'weight': 0.4,
            'uk': {'weight': 1.},
        }
    },

    'real_estate': {
        'weight': 0.1,
        'global': {'weight': 1.},
    },
}
