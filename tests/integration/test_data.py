from alphadog.data.retrieval import PriceData


def test_price_data():
    """End-to-end test to check that PriceData can retrieve data."""
    prices = PriceData.from_instrument_id("UKCORP")
    assert not prices.df.empty
