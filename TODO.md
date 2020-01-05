# TODO
Keep track of ongoing work and priorities.


## Data Sources
- yfinance
- quandl
- eoddata?


## Plan
- Create strategies to contain the parameterised versions of the signal that will be run.
  Longer term there will be a framework to run each of these as metrics on some data fixtures.
- Implement the Forecast class for a momentum strategy with 3 speeds
- Implement MarketData class. Think about design. Inherit from a generic BaseData class so that
  later FundamentalData, MacroData etc will be consistent?
- Implement InstrumentForecast class
- Implement Portfolio class
- Consider adding an Instrument class which for each instrument contains: 
  long_name, asset_class, currency, p_weight?, market_data call and params
- data_retrieval should handle periodic/daily retrieval of new data nd save to some database
  (CSV file for now?). The signals should all read their data from this database. 
- Get data for FINRA short volume - see email
- Get data from quandl
- Bulk out technical indicators using ta-lib
  
  
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



## To Read:
- Kelly Criterion
- Perry Kaufman signals book
- Leveraged trading


## Future modules:
- evaluation - maybe roll this into portfolio_management
- execution - once trading is automated
- risk