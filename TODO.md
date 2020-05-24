# TODO
Keep track of ongoing work and priorities.


## Data Sources
- yfinance
- quandl
- FINRA
- eoddata?


## Plan
- Add tests for TODO TEST placeholders
- Flesh out portfolio framework and implement missing functionality
- Try out some mean reversion strategies - from Algorithmic Trading
- Add trading_capital csv, Handle reindexing trading_capital, calculating new capital to ffill etc
- Add discretionary forecasts to data.inputs - date, instrument_id, forecast
- Add accounting module - see pysystemtrade and quantopian pyfolio and alphalens
- Bulk out technical indicators using ta-lib
- Backfill FINRA data to 2010
- Get data from quandl
- Consider mongodb/arctic for data storage?
- Alter weights based on bootstrapped backtests - run once as research and hard code these? Also
  handle missing items correctly
  
  
## Done
- Create first strategy - simple ewma momentum
- Refactor strategies as signals, which will have the raw signal functions.
- Create framework folder to handle forecast scaling, diversification multipliers, weights etc.
- Create a second strategy (mean reversion? open to close?) to implement in the InstrumentForecast
  class. Created breakout signal.
- Add tests for breakout
- Choose instruments
- Create instruments_config.json
- Get data from yfinance 
- data retrieval should handle periodic/daily retrieval of new data and save to some database
  (CSV file for now?). The signals should all read their data from this database. 
- Add a long bias signal which is a constant 10
- WON'T DO. Create strategies to contain the parameterised versions of the signal that will be run.
  Longer term there will be a framework to run each of these as metrics on some data fixtures.
- Add intermittent nans test to trend signals
- Write parser for FINRA short volume - see email
- Create a flat config for and then import this into a hierarchy instead of directly specifying both 
- Outline the Forecast class
- Outline InstrumentForecast class
- Outline Portfolio class
- Create signals_config.json - for each strategy specify the parameters it will run with, the forecast scalar
- Consider adding an Instrument class which for each instrument contains: 
  long_name, asset_class, currency, p_weight?, market_data call and params
- Implement MarketData class. Think about design. Inherit from a generic BaseData class so that
  later FundamentalData, MacroData etc will be consistent?
- FX data:
-- Get FX data - GBP, GBX, EUR, USD - add these to the yfinance symbol mapping and run a backfill
-- Write get_fx function in data.retrieval - PriceData.from_instrument_id(fx_symbol)
-- Add fx_rate property to Subsystem
- Implement vol_scalar
- Add some tests and tidy unused/empty modules
- Calculate weights. Manually assign hierarchy weights in config. 
- Handle passing different data objects from required_data_fixtures.
  Constant dict which maps each fixture name to the data retrieval function.
- Add integration test - run everything, check we get something. Use scratch_5 as a starting point.
- Read Algorithmic Trading, Ernie Chan
- Calc returns function - pct, log, diff
- Fix vol_scalar function
- Add subsystem_returns property to Subsystem
- Fix Portfolio combine_subsystems - calc div_multiplier using Subsystem returns
- Add tests for framework module: add Portfolio tests.
- Make repr nice for  Portfolio, Subsystem, Forecast


## To Read:
- Kelly Criterion
- Perry Kaufman signals book
- Leveraged trading


## Future modules:
- accounting - maybe roll this into portfolio_management
- execution - once trading is automated
- risk - submodule of analytics?