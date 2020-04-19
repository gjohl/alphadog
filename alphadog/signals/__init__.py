"""
Raw functionality which make up strategies once parameterised.

The naming convention is starts the signal name with:
- "V": univariate signals on price data
- "X": cross-sectional signals
- "B": bias signals for long/short tilts
- "F": signals using fundamental data

-- Momentum - VMOM
-- Breakout - VBO
-- Bias - BLONG, BSHORT
-- carry
-- value
-- fundamental
"""
from alphadog.signals.trend import momentum_signal, breakout_signal
from alphadog.signals.bias import long_bias_signal, short_bias_signal

PARAMETERISED_SIGNALS = {}

# Momentum signals
for speed in range(1, 7):
    fast = 2 ** speed
    slow = 4 * fast
    sig = lambda df: momentum_signal(df, fast, slow)
    PARAMETERISED_SIGNALS[f"VMOM{speed}"] = sig

# Breakout signals
for speed in range(1, 7):
    lookback = 10 * 2 ** (speed-1)
    sig = lambda df: breakout_signal(df, lookback)
    PARAMETERISED_SIGNALS[f"VBO{speed}"] = sig

# Bias signals
PARAMETERISED_SIGNALS["BLONG"] = long_bias_signal
PARAMETERISED_SIGNALS["BSHORT"] = short_bias_signal

