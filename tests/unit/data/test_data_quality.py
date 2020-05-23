from alphadog.data.data_quality import (
    staleness, check_nonempty_dataframe, check_price_data
)


class TestStaleness:

    def test_success(self):
        # TODO TEST
        pass

    def test_empty_df_raises(self):
        # TODO TEST
        pass

    def test_non_date_index(self):
        # TODO TEST
        pass


class TestCheckNonemptyDataframe:

    def test_empty_df(self):
        # TODO TEST
        pass

    def test_df_all_nan(self):
        # TODO TEST
        pass

    def test_success(self):
        # TODO TEST
        pass


class TestNonnegativeScalar:

    def test_positive_scalar(self):
        # TODO TEST
        pass

    def test_negative_scalar_raises(self):
        # TODO TEST
        pass



class TestCheckPriceData:

    def test_empty_df(self):
        # TODO TEST
        pass

    def test_df_all_nan(self):
        # TODO TEST
        pass

    def test_bad_columns_raises(self):
        # TODO TEST
        pass

    def test_bad_prices_raises(self):
        # TODO TEST
        pass

    def test_success(self):
        # TODO TEST
        pass
