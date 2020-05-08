import click

from alphadog.data.yfinance_data import update_yfinance_data
from alphadog.data.constants import YFINANCE_SYMBOLS


@click.command()
@click.option("--symbols", default=YFINANCE_SYMBOLS, help="Symbols to update.")
def run(symbols=YFINANCE_SYMBOLS):
    for symbol in symbols:
        update_yfinance_data(symbol)
    return


if __name__ == "__main__":
    # TODO: use argparse to pass arguments and run this as from the command line
    run()
