import numpy as np
import pytest

from alphadog.internals.fx import get_fx


class TestGetFx:

    def test_success(self):
        actual = get_fx('GBP', 'USD')
        assert actual['GBPUSD'].min() > 1
        assert actual['GBPUSD'].max() < 3
        assert actual.shape[0] > 0
        assert actual.shape[0] > 0
        assert actual.shape[1] == 1
        assert all(actual.columns == ['GBPUSD'])

    @pytest.mark.parametrize('ccy', ['GBP', 'GBX', 'EUR', 'USD', 'ABC'])
    def test_no_conversion(self, ccy):
        actual = get_fx(ccy, ccy)
        assert actual == 1

    def test_gbp_to_gbx(self):
        actual = get_fx('GBP', 'GBX')
        assert actual == 100.

    def test_gbx_to_gbp(self):
        actual = get_fx('GBX', 'GBP')
        assert actual == 0.01

    @pytest.mark.parametrize('to_ccy', ['EUR', 'USD'])
    def test_gbx_to_other_ccy(self, to_ccy):
        gbp_rate = get_fx('GBP', to_ccy)
        gbx_rate = get_fx('GBX', to_ccy)
        assert all(np.isclose(gbp_rate / gbx_rate, 100))

    @pytest.mark.parametrize('from_ccy', ['EUR', 'USD'])
    def test_other_ccy_to_gbx(self, from_ccy):
        gbp_rate = get_fx(from_ccy, 'GBP')
        gbx_rate = get_fx(from_ccy, 'GBX')
        assert all(np.isclose(gbp_rate / gbx_rate, 0.01))
