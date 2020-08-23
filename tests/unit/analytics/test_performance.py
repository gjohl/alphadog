import numpy as np
import pandas as pd
import pytest

from alphadog.analytics.performance import returns, arithmetic_returns, geometric_returns


@pytest.fixture
def expected_arithmetic_price_returns():
    return pd.DataFrame(
        data=[np.nan, 0.025, 0.024390243902439046, 0.023809523809523725,
              0.023255813953488413, 0.022727272727272707, 0.022222222222222143,
              0.021739130434782705, 0.02127659574468077, 0.02083333333333326,
              0.020408163265306145, 0.02, 0.019607843137254832,
              0.019230769230769162, 0.018867924528301883, 0.0185185185185186,
              0.018181818181818077, 0.017857142857142794, 0.01754385964912286,
              0.01724137931034475],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['price']
    )


@pytest.fixture
def expected_geometric_price_returns():
    return pd.DataFrame(
        data=[np.nan, 0.0246926125903717, 0.02409755157906046, 0.023530497410193973,
              0.022989518224698635, 0.022472855852058604, 0.02197890671877545,
              0.02150620522096336, 0.021053409197832718, 0.020619287202735315,
              0.020202707317519497, 0.019802627296179764, 0.019418085857101808,
              0.019048194970694432, 0.018692133012152556, 0.018349138668196652,
              0.018018505502678472, 0.01769957709940062, 0.01739174271186883,
              0.01709443335930061],
        index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp'),
        columns=['price']
    )


class TestCalcReturns:

    @pytest.mark.parametrize('percent, multiplier', [[False, 1.], [True, 100.]])
    def test_arithmetic(self, mock_price, expected_arithmetic_price_returns, percent, multiplier):
        actual = returns(mock_price, 'arithmetic', percent)
        expected = expected_arithmetic_price_returns * multiplier
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('percent, multiplier', [[False, 1.], [True, 100.]])
    def test_geometric(self, mock_price, expected_geometric_price_returns, percent, multiplier):
        actual = returns(mock_price, 'geometric', percent)
        expected = expected_geometric_price_returns * multiplier
        pd.testing.assert_frame_equal(actual, expected)

    def test_invalid_return_type_raises(self, mock_price):
        return_type = "INVALID"
        expected_msg = "return_type must be 'arithmetic' or 'geometric'. Got INVALID"
        with pytest.raises(NotImplementedError, match=expected_msg):
            returns(mock_price, return_type)


class TestCalcArithmeticReturns:

    def test_single_column(self, mock_price, expected_arithmetic_price_returns):
        actual = arithmetic_returns(mock_price)
        pd.testing.assert_frame_equal(actual, expected_arithmetic_price_returns)

    def test_multi_column(self, mock_ohlcv):
        actual = arithmetic_returns(mock_ohlcv)
        expected = pd.DataFrame(
            {'open': [np.nan, 0.025, 0.024390243902439046, 0.023809523809523725,
                      0.023255813953488413, 0.022727272727272707, 0.022222222222222143,
                      0.021739130434782705, 0.02127659574468077, 0.02083333333333326,
                      0.020408163265306145, 0.020000000000000018, 0.019607843137254832,
                      0.019230769230769162, 0.018867924528301883, 0.0185185185185186,
                      0.018181818181818077, 0.017857142857142794, 0.01754385964912286,
                      0.01724137931034475],
             'high': [np.nan, 0.023809523809523725, 0.023255813953488413, 0.022727272727272707,
                      0.022222222222222143, 0.021739130434782705, 0.02127659574468077,
                      0.02083333333333326, 0.020408163265306145, 0.020000000000000018,
                      0.019607843137254832, 0.019230769230769162, 0.018867924528301883,
                      0.0185185185185186, 0.018181818181818077, 0.017857142857142794,
                      0.01754385964912286, 0.01724137931034475, 0.016949152542372836,
                      0.016666666666666607],
             'low': [np.nan, 0.02564102564102555, 0.02499999999999991, 0.024390243902439046,
                     0.023809523809523725, 0.023255813953488413, 0.022727272727272707,
                     0.022222222222222143, 0.021739130434782705, 0.02127659574468077,
                     0.02083333333333326, 0.020408163265306145, 0.020000000000000018,
                     0.019607843137254832, 0.019230769230769162, 0.018867924528301883,
                     0.0185185185185186, 0.018181818181818077, 0.017857142857142794,
                     0.01754385964912286],
             'close': [np.nan, 0.024390243902439046, 0.023809523809523725, 0.023255813953488413,
                       0.022727272727272707, 0.022222222222222143, 0.021739130434782705,
                       0.02127659574468077, 0.02083333333333326, 0.020408163265306145,
                       0.02, 0.019607843137254832, 0.019230769230769162, 0.018867924528301883,
                       0.0185185185185186, 0.018181818181818077, 0.017857142857142794,
                       0.01754385964912286, 0.01724137931034475, 0.016949152542372836],
             'volume': [np.nan, 0.025, 0.024390243902439046, 0.023809523809523725,
                        0.023255813953488413, 0.022727272727272707, 0.022222222222222143,
                        0.021739130434782705, 0.02127659574468077, 0.02083333333333326,
                        0.020408163265306145, 0.020000000000000018, 0.019607843137254832,
                        0.019230769230769162, 0.018867924528301883, 0.0185185185185186,
                        0.018181818181818077, 0.017857142857142794, 0.01754385964912286,
                        0.01724137931034475]},
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp')
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_with_intermittent_nans(self, mock_price, expected_arithmetic_price_returns):
        price_df = mock_price.copy()
        price_df.iloc[[5, 15], :] = np.nan
        actual = arithmetic_returns(price_df)
        expected = expected_arithmetic_price_returns.copy()
        expected.iloc[[5, 6, 15, 16], :] = np.nan
        pd.testing.assert_frame_equal(actual, expected)
        pass

    def test_empty_dataframe(self, mock_ohlcv):
        ohlcv_df = mock_ohlcv.head(0)
        actual = arithmetic_returns(ohlcv_df)
        expected = pd.DataFrame(data=[], index=pd.DatetimeIndex([], name='timestamp'),
                                columns=['open', 'high', 'low', 'close', 'volume'],
                                dtype=float)
        pd.testing.assert_frame_equal(actual, expected)

    def test_series(self, mock_price, expected_arithmetic_price_returns):
        price_series = mock_price.iloc[:, 0]
        actual = arithmetic_returns(price_series)
        expected = expected_arithmetic_price_returns.iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected)

    def test_empty_series(self, mock_price):
        price_series = mock_price.iloc[:, 0].head(0)
        actual = arithmetic_returns(price_series)
        expected = pd.Series(data=[], index=pd.DatetimeIndex([], name='timestamp'), name='price')
        pd.testing.assert_series_equal(actual, expected)


class TestCalcGeometricReturns:

    def test_single_column(self, mock_price, expected_geometric_price_returns):
        actual = geometric_returns(mock_price)
        pd.testing.assert_frame_equal(actual, expected_geometric_price_returns)

    def test_multi_column(self, mock_ohlcv):
        actual = geometric_returns(mock_ohlcv)
        expected = pd.DataFrame(
            {'open': [np.nan, 0.0246926125903717, 0.02409755157906046, 0.023530497410193973,
                      0.022989518224698635, 0.022472855852058604, 0.02197890671877545,
                      0.02150620522096336, 0.021053409197832718, 0.020619287202735315,
                      0.020202707317519497, 0.019802627296179764, 0.019418085857101808,
                      0.019048194970694432, 0.018692133012152556, 0.018349138668196652,
                      0.018018505502678472, 0.01769957709940062, 0.01739174271186883,
                      0.01709443335930061],
             'high': [np.nan, 0.023530497410193973, 0.022989518224698635, 0.022472855852058604,
                      0.02197890671877545, 0.02150620522096336, 0.021053409197832718,
                      0.020619287202735315, 0.020202707317519497, 0.019802627296179764,
                      0.019418085857101808, 0.019048194970694432, 0.018692133012152556,
                      0.018349138668196652, 0.018018505502678472, 0.01769957709940062,
                      0.01739174271186883, 0.01709443335930061, 0.01680711831638071,
                      0.01652930195121094],
             'low': [np.nan, 0.025317807984289953, 0.0246926125903717, 0.02409755157906046,
                     0.023530497410193973, 0.022989518224698635, 0.022472855852058604,
                     0.02197890671877545, 0.02150620522096336, 0.021053409197832718,
                     0.020619287202735315, 0.020202707317519497, 0.019802627296179764,
                     0.019418085857101808, 0.019048194970694432, 0.018692133012152556,
                     0.018349138668196652, 0.018018505502678472, 0.01769957709940062,
                     0.01739174271186883],
             'close': [np.nan, 0.02409755157906046, 0.023530497410193973, 0.022989518224698635,
                       0.022472855852058604, 0.02197890671877545, 0.02150620522096336,
                       0.021053409197832718, 0.020619287202735315, 0.020202707317519497,
                       0.019802627296179764, 0.019418085857101808, 0.019048194970694432,
                       0.018692133012152556, 0.018349138668196652, 0.018018505502678472,
                       0.01769957709940062, 0.01739174271186883, 0.01709443335930061,
                       0.01680711831638071],
             'volume': [np.nan, 0.024692612590371255, 0.024097551579060905, 0.023530497410193973,
                        0.022989518224699523, 0.02247285585205816, 0.02197890671877545,
                        0.02150620522096247, 0.02105340919783316, 0.020619287202736203,
                        0.020202707317519497, 0.019802627296179764, 0.01941808585710092,
                        0.019048194970693544, 0.01869213301215389, 0.018349138668195764,
                        0.01801850550267936, 0.017699577099399733, 0.01739174271186883,
                        0.01709443335930061]},
            index=pd.DatetimeIndex(pd.bdate_range('2019-01-01', periods=20), name='timestamp')
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_with_intermittent_nans(self, mock_price, expected_geometric_price_returns):
        price_df = mock_price.copy()
        price_df.iloc[[5, 15], :] = np.nan
        actual = geometric_returns(price_df)
        expected = expected_geometric_price_returns.copy()
        expected.iloc[[5, 6, 15, 16], :] = np.nan
        pd.testing.assert_frame_equal(actual, expected)

    def test_empty_dataframe(self, mock_ohlcv):
        ohlcv_df = mock_ohlcv.head(0)
        actual = geometric_returns(ohlcv_df)
        expected = pd.DataFrame(data=[], index=pd.DatetimeIndex([], name='timestamp'),
                                columns=['open', 'high', 'low', 'close', 'volume'],
                                dtype=float)
        pd.testing.assert_frame_equal(actual, expected)

    def test_series(self, mock_price, expected_geometric_price_returns):
        price_series = mock_price.iloc[:, 0]
        actual = geometric_returns(price_series)
        expected = expected_geometric_price_returns.iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected)

    def test_empty_series(self, mock_price):
        price_series = mock_price.iloc[:, 0].head(0)
        actual = geometric_returns(price_series)
        expected = pd.Series(data=[], index=pd.DatetimeIndex([], name='timestamp'), name='price')
        pd.testing.assert_series_equal(actual, expected)


# TODO
class TestRobustVolatility:

    def test_success(self):
        pass

    def test_no_rolling_floor(self):
        pass

    def test_backfill(self):
        pass

    def test_empty_input(self):
        pass
