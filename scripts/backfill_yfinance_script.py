from alphadog.data.yfinance_data import backfill_yfinance_data
from alphadog.data.constants import YFINANCE_SYMBOLS


def run(symbols=YFINANCE_SYMBOLS):
    for symbol in symbols:
        backfill_yfinance_data(symbol)
    return


if __name__ == "__main__":
    # TODO: use argparse to pass arguments and run this as from the command line
    run()
