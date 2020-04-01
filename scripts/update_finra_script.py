"""
Update the given range of data for FINRA short volumes.

Fills the date from the most recent date saved on disk until the most recent day of data available.
"""
# TODO
import click
import re
import os

from alphadog.data.finra_data import backfill_finra_data
from alphadog.data.constants import FINRA_EXCHANGES, DATA_DIR, FINRA_DIR


START_DATE = "2020-01-01"


@click.command()
@click.option("--exchanges", default=FINRA_EXCHANGES, help="Exchange codes to run.")
def run(exchanges):
    # Get most recent day of data for CNMS
    filepath = os.path.join(DATA_DIR, FINRA_DIR, 'CNMS')
    file_list = sorted(os.listdir(filepath))
    latest_file = file_list[-1]
    start_date = re.findall('20[0-9]{6}', latest_file)[0]

    # Backfill from that date until today
    backfill_finra_data(start_date, None, exchanges)
    return


if __name__ == "__main__":
    run()
