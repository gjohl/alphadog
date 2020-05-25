"""
Strategies are parameterised signals ready to run on a data fixture.

The naming convention is to start the signal name with:
- "V": univariate signals on price data
- "X": cross-sectional signals
- "B": bias signals for long/short tilts
- "F": signals using fundamental data

This config defines the components of the strategies, in the following format.
{
    'VMOM1': {
        'signal_func': momentum_signal,
        'required_data_fixtures': ['price_df'],
        'params': {'fast': 2, 'slow': 8},
        'strategy_name': 'VMOM1'
    }
}

Data fixtures are denoted by names which correspond to data retrieval functions.
Currently, these data retrieval functions each take an instrument_id as its argument
and return a DataFrame of data.
{fixture_name: retrieval_function}

# TODO:
In future, this could be made more flexible if the arg is too restrictive,
instead making required data fixtures a dict of
{fixture_name: params}

FIXME:
The signal funcs must be lambdas with KEYWORD ARGS due to this bug:
https://stackoverflow.com/questions/25670516/strange-overwriting-occurring-when-using-lambda-functions-as-dict-values

"""
from alphadog.data.retrieval import PriceData
from alphadog.signals.trend import momentum_signal, breakout_signal
from alphadog.signals.bias import long_bias_signal


##############
# Strategies #
##############
PARAMETERISED_STRATEGIES = {}

# Momentum signals
for speed in range(1, 7):
    fast = 2 ** speed
    slow = 4 * fast
    sig = lambda price_df, fast=fast, slow=slow: momentum_signal(price_df, fast, slow)  # noqa: E731
    strategy_name = f"VMOM{speed}"

    temp_dict = {'signal_func': sig,
                 'raw_signal_func': momentum_signal,
                 'required_data_fixtures': ['price_df'],
                 'params': {'fast': fast, 'slow': slow},
                 'strategy_name': strategy_name,
                 'hierarchy_1': 'trend',
                 'hierarchy_2': 'VMOM'}
    PARAMETERISED_STRATEGIES[strategy_name] = temp_dict

# Breakout signals
for speed in range(1, 7):
    lookback_period = 10 * 2 ** (speed-1)
    sig = (lambda price_df, lookback_period=lookback_period:
           breakout_signal(price_df, lookback_period))
    strategy_name = f"VBO{speed}"

    temp_dict = {'signal_func': sig,
                 'raw_signal_func': breakout_signal,
                 'required_data_fixtures': ['price_df'],
                 'params': {'lookback_period': lookback_period},
                 'strategy_name': strategy_name,
                 'hierarchy_1': 'trend',
                 'hierarchy_2': 'VBO'}
    PARAMETERISED_STRATEGIES[strategy_name] = temp_dict

# Bias signals
PARAMETERISED_STRATEGIES["BLONG"] = {
    'signal_func': long_bias_signal,
    'raw_signal_func': long_bias_signal,
    'required_data_fixtures': ['price_df'],
    'params': {},
    'strategy_name': "BLONG",
    'hierarchy_1': 'bias',
}

# PARAMETERISED_STRATEGIES["BSHORT"] = {
#     'signal_func': short_bias_signal,
#     'raw_signal_func': short_bias_signal,
#     'required_data_fixtures': ['price_df'],
#     'params': {},
#     'strategy_name': "BSHORT",
#     'hierarchy_1': 'bias',
# }


#################
# Data Fixtures #
#################

# TODO: move to data.retrieval?
def get_price_df(instrument_id):
    return PriceData.from_instrument_id(instrument_id).df


DATA_FIXTURES = {
    'price_df': get_price_df
}
