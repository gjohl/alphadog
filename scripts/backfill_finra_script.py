"""
Backfill the given range of data for FINRA short volumes.
"""
import click

from alphadog.data.finra_data import backfill_finra_data
from alphadog.data.constants import FINRA_EXCHANGES


START_DATE = "2020-01-01"


@click.command()
@click.option("--start_date", default=START_DATE, help="Start of date range.")
@click.option("--end_date", default=None, help="End of date range.")
@click.option("--exchanges", default=FINRA_EXCHANGES, help="Exchange codes to run.")
def run(start_date, end_date, exchanges):
    backfill_finra_data(start_date, end_date, exchanges)
    return


if __name__ == "__main__":
    run()
