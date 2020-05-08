import pytest
import pandas as pd

from alphadog.data.retrieval import (
    BaseData, PriceData
)


@pytest.mark.parametrize('input_name', ['test_name', None])
def test_base_data(input_name, mock_ohlcv):
    actual = BaseData(mock_ohlcv, input_name)
    pd.testing.assert_frame_equal(actual.df, mock_ohlcv)
    assert actual.name == input_name
    assert hasattr(actual, 'staleness')
    assert actual.__repr__() == f"{input_name} <class 'alphadog.data.retrieval.BaseData'>"


@pytest.mark.parametrize('input_name', ['test_name', None])
def test_base_data_from_dataframe(input_name, mock_ohlcv):
    actual = BaseData.from_dataframe(mock_ohlcv, input_name)
    pd.testing.assert_frame_equal(actual.df, mock_ohlcv)
    assert actual.name == input_name
    assert hasattr(actual, 'staleness')
    assert actual.__repr__() == f"{input_name} <class 'alphadog.data.retrieval.BaseData'>"


@pytest.mark.parametrize('input_name', ['test_name', None])
def test_price_data(input_name, mock_ohlcv):
    input_df = mock_ohlcv[['close']]
    actual = PriceData(input_df, input_name)
    pd.testing.assert_frame_equal(actual.df, input_df)
    assert actual.name == input_name
    assert hasattr(actual, 'staleness')
    assert actual.__repr__() == f"{input_name} <class 'alphadog.data.retrieval.PriceData'>"
