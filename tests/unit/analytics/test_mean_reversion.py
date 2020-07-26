from numpy import cumsum, log
from numpy.random import randn, seed
import pandas as pd
import pytest

# TODO TEST: all of these tests
MOCK_NUM_OBS = 1000


@pytest.fixture
def mock_geometric_brownian_motion_df():
    seed(1)
    values = log(cumsum(randn(MOCK_NUM_OBS)) + 1000)
    df = pd.DataFrame(data=values,
                      columns=['close'],
                      index=pd.bdate_range(start="2019-01-01", periods=MOCK_NUM_OBS))
    return df


@pytest.fixture
def mock_mean_reverting_df():
    seed(1)
    values = log(randn(MOCK_NUM_OBS)+1000)
    df = pd.DataFrame(data=values,
                      columns=['close'],
                      index=pd.bdate_range(start="2019-01-01", periods=MOCK_NUM_OBS))
    df = df.ewm(span=10).mean()
    return df


@pytest.fixture
def mock_trending_df():
    seed(1)
    values = log(cumsum(randn(MOCK_NUM_OBS)+1)+1000)
    df = pd.DataFrame(data=values,
                      columns=['close'],
                      index=pd.bdate_range(start="2019-01-01", periods=MOCK_NUM_OBS))
    return df


class TestAugmentedDickeyFuller:

    def test_mean_reverting_input(self):
        pass

    def test_trending_input(self):
        pass

    def test_brownian_motion_input(self):
        pass

    def test_empty_input(self):
        pass


class TestHurstExponent:

    def test_mean_reverting_input(self):
        pass

    def test_trending_input(self):
        pass

    def test_brownian_motion_input(self):
        pass

    def test_empty_input(self):
        pass

    def test_num_obs_less_than_min_lags_raises(self):
        pass


class TestVarianceRatioTest:

    def test_mean_reverting_input(self):
        pass

    def test_trending_input(self):
        pass

    def test_brownian_motion_input(self):
        pass

    def test_empty_input(self):
        pass


class TestMeanReversionHalfLife:

    def test_mean_reverting_input(self):
        pass

    def test_trending_input(self):
        pass

    def test_brownian_motion_input(self):
        pass

    def test_empty_input(self):
        pass

    def test_positive_slope_raises(self):
        # If not already covered in the tests above
        pass


class TestCointegratedAugmentedDickeyFuller:

    def test_mean_reverting_input(self):
        pass

    def test_trending_input(self):
        pass

    def test_brownian_motion_input(self):
        pass

    def test_empty_input(self):
        pass


class TestJohansenTest:

    def test_two_time_series(self):
        pass

    def test_several_time_series(self):
        pass

    def test_invalid_significance_level_raises(self):
        pass

    def test_invalid_method_input_raises(self):
        pass
