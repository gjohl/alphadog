import click

from alphadog.data.yfinance_data import backfill_yfinance_data
from alphadog.data.constants import YFINANCE_SYMBOLS


@click.command()
@click.option("--symbols", default=YFINANCE_SYMBOLS, help="Symbols to backfill.", type=list)
def run(symbols=YFINANCE_SYMBOLS):
    for symbol in symbols:
        backfill_yfinance_data(symbol)
    return


if __name__ == "__main__":
    run()
