import os

from alphadog.constants import PROJECT_DIR


DATA_DIR = os.path.join(PROJECT_DIR, "alphadog/data/csv_files/")

# Yahoo Finance
PRICE_COLS = {'close'}
OHLCV_COLS = {'open', 'high', 'low', 'close', 'volume'}
YFINANCE_DIR = "yfinance/"
YFINANCE_REQUIRED_COLS = {'open', 'high', 'low', 'close', 'volume'}
YFINANCE_SYMBOL_INSTRUMENT_ID_MAPPING = {
    "^FTSE": "FTSE100",
    "^FTMC": "FTSE250",
    "^FTAS": "FTSEAS",
    "^GSPC": "SP500",
    "VGK": "FTSEEURO",
    "VGOV.L": "UKGOV",
    "IE00BFRTDB69.IR": "USGOV",
    "GOVY.L": "EUROGOV",
    "IE00B1S74W91.IR": "UKCORP",
    "LQD": "GLOBALCORP",
    "INXG.L": "UKILB",
    "FFR": "GLOBALREIT",
    "GBPUSD=X": "GBPUSD",
    "USDGBP=X": "USDGBP",
    "EURUSD=X": "EURUSD",
    "USDEUR=X": "USDEUR",
    "GBPEUR=X": "GBPEUR",
    "EURGBP=X": "EURGBP",
}
YFINANCE_SYMBOLS = set(YFINANCE_SYMBOL_INSTRUMENT_ID_MAPPING.keys())

# FINRA Short Volume
FINRA_DIR = "finra/"
FINRA_BASE_URL = 'http://regsho.finra.org/'
FINRA_EXCHANGES = ['CNMS', 'FNQC', 'FNRA', 'FNSQ', 'FNYX', 'FORF']
