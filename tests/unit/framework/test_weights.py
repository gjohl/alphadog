import numpy as np
import pytest

from alphadog.framework.weights import STRATEGY_WEIGHTS, INSTRUMENT_WEIGHTS


@pytest.mark.parametrize('input_config', [STRATEGY_WEIGHTS, INSTRUMENT_WEIGHTS])
def test_l1_weights_sum_to_unity(input_config):
    l1_weights = []
    for l1_dicts in input_config.values():
        l1_weights.append(l1_dicts['weight'])

    assert np.isclose(sum(l1_weights), 1)
