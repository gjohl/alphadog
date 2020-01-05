DATA_DIR = "/home/gurp/workspace/alphadog/alphadog/data/csv_files/"

# Yahoo Finance
YFINANCE_DIR = "yfinance_csv/"
YFINANCE_REQUIRED_COLS = {'open', 'high', 'low', 'close', 'volume'}
YFINANCE_SYMBOLS = {"^FTSE", "^FTMC", "^FTAS", "^GSPC", "AWNT05UK.L", "VGOV.L", "IE00BFRTDB69.IR",
                    "GOVY.L", "IE00B1S74W91.IR", "LQD", "INXG.L", "FFR"}
YFINANCE_SYMBOL_INSTRUMENT_ID_MAPPING = {
    "^FTSE": "FTSE100",
    "^FTMC": "FTSE250",
    "^FTAS": "FTSEAS",
    "^GSPC": "SP500",
    "AWNT05UK.L": "FTSEEURO",
    "VGOV.L": "UKGOV",
    "IE00BFRTDB69.IR": "USGOV",
    "GOVY.L": "EUROGOV",
    "IE00B1S74W91.IR": "UKCORP",
    "LQD": "GLOBALCORP",
    "INXG.L": "UKILB",
    "FFR": "GLOBALREIT"
}


