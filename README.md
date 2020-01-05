# Alphadog

Module for systematic trading signals.


## Instruments
Start with ETFs, then try single stocks.
Read leveraged trading for ideas on CFDs and spread betting.

Eventually the multi-asset portfolio will include:
- Equity
- Bonds
- Commodities
- Real Estate
- Crypto

To begin with, we'll get the instruments we will use are ETFs or funds for equity and bonds, with some mix of geographies to build out the multi-level portfolio functionality of the framework.

The general framework will follow a hierarchy as below, with some variations depending on the asset class.
For example, cryptocurrencies don't make sense to split by country.
- Asset class
-- Country
--- Sector

Initially this will be as follows. 

- Equity
-- UK
--- FTSE 100
--- FTSE 250
--- FTSE All Shares
-- US
-- Eurostoxx

- Bonds
-- Gov
--- UK
--- US
--- Europe
-- Corporate
--- UK
--- Global
-- Inflation-linked

- Real Estate
-- Global


The criteria for choosing an instrument are:
- Data available from yfinance (and preferably also quandl)
- Big, liquid instrument
- Low trading costs. For now, as an MVP, this will be judged by the cost of trading in a
  Hargreaves Lansdowne account but later I'll make other broker accounts if it is worthwhile.


## Forecasts
In units of risk-adjusted returns (Sharpe ratio).
Scale so that average long signal is +10, average short is -10.
Spline extreme values to +-20. Think about scaling back to 10/15 at very extreme values.
AVG_FORECAST = 10

Allow discretionary forecasts to have a section of the strategy.
Rate a security on this +-20 scale.

### Adding new trading rules
- Convert ideas into a continuous rule (not a binary buy/sell).
- Vol scale so that forecasts are comparable across markets and data can be pooled.
- Find the forecast scalar by looking at the distribution of historical forecasts
  (without looking at backtested _performance_) and setting the scalar so that the average absolute
  value is 10.



## Combined forecasts
Handcraft weights (or bootstrap based on historical covariances and expected returns.

Level 1 - split by trading rule, e.g. ewmac vs carry
Level 2 - split by variations, e.g momentum speeds. Consider correlations, middle speed is the
          most correlated of the 3 so downweight, see handcrafted weights

Forecast diversification multiplier to get average absolute combined forecast back to 10.
Cap this at 2.5.

Spline again.



## Volatility targeting
Half-Kelly criterion.
Rule of thumb - vol target is half of expected Sharpe ratio
Negative skew makes expected losses worse than vol would suggest
Higher mean makes expected losses better than vol suggests.

Need leverage to achieve this vol target.
Exclude low vol instruments.
Deflate the backtested Sharpe to get expected Sharpe, according to page 100.
Adjust the vol target down further if the strategy has negative skew.

Start with constant vol target and low capital, and slowly increase the capital amount as one
gains confidence.

Percentage vol target is constant. Cash volatility target varies depending on performance.
If the portfolio has £100k and 30% vol taget, cash vol is £30k. If the portfolio drops 10%,
the cash vol target should be adjusted to £90k * 30% = £27k. This is "rolling up profits and
losses".

pct_vol_target_annualised = annualised expected vol
trading_capital = £ available to invest
cash_vol_target_annualised = vol_target_pct * trading_capital
cash_vol_target_daily = cash_vol_annualised / sqrt(252)



## Position Sizing
### Instrument Block
An instrument block is defined "one unit of that instrument". The block value is "how much you gain
or lose if the instrument's price changes by 1%".
 This varies by asset class:
- Equity - For lot sizes of 100 shares this is simply the share price, because
           100 * 1% of share price = share price
- Futures - 1% of price * contract size (e.g. number of barrels of crude per contract)
- FX futures - Price is 100 - interest rate. Account for quarterly contracts with day count.
               Notional * 1% of price * (days/365)

### Price Volatility
The above considers a 1% price move but doesn't account for the likelihood of a 1% price move.
More volatile instruments are more likely to experience a 1% move, so vol normalise again.
Measure price volatility: the standard deviation of prices of some lookback. 5 weeks (25 days) is
a standard lookback period.

### Instrument Currency Volatility
The expected standard deviation of daily returns from owning one instrument block in the currency
of the instrument.
instrument_currency_vol = block_value * price_vol

### Instrument Value Volatility
The instrument may not be in the same currency as your trading account, so convert to the
currency of the trading account. Make sure the currency conversion is the correct way around.
instrument_value_vol = instrument_currency_vol * fx_rate

### Position
Instrument value volatility gives the vol contribution per block of the instrument.
Ignoring forecasts for the moment, the vol_scalar gives the number of blocks to buy to achieve the
vol target using this instrument alone.
vol_scalar = cash_vol_target_daily / instrument_value_vol

Now considering the forecast value, recall that the forecast value has been scaled such that the
average value is 10. If our forecast is 10, the vol_scalar above matches our desired position
exactly; if our forecast is higher our position should be higher and vice versa.
subsystem_position = vol_scalar * (instrument_forecast / AVG_FORECAST)



## Portfolios
### Instrument weights
Instrument weights determine how much to allocate to each trading subsystem.
Do this from handcrafting or bootstrapping.

Need the correlations between trading subsystem returns, not the correlations between instrument
returns. As an approximation, the correlation between subsystem returns is 0.7 of the correlation
of instrument returns.

### Instrument diversification multiplier
Apply an instrument diversification multiplier to account for the decrease in vol from diversifying across instruments. Calculate this as 1/sqrt(WHWt). See p 297.

Limit the value of the multiplier to 2.5.

### Portfolio instrument position
= Subsystem position * instrument weight * instrument diversification multiplier

### Generating trades
Rounded_target_position = round(portfolio_instrument_position)

If this is within 10% of the current position, make no change. This is position inertia or buffering.
If more than 10% away from target, trade to the target.


