"""
Example usage of the trading indicators library.
"""

import pandas as pd
import numpy as np
from trading_indicators import sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator

# Sample price data
prices = pd.Series([
    100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
    111, 113, 112, 114, 116, 115, 117, 119, 118, 120
])

print("=" * 60)
print("Trading Indicators Example")
print("=" * 60)

# Simple Moving Average
print("\n1. Simple Moving Average (5-period)")
sma_values = sma(prices, 5)
print(f"Last 5 values: {sma_values.tail(5).values}")

# Exponential Moving Average
print("\n2. Exponential Moving Average (5-period)")
ema_values = ema(prices, 5)
print(f"Last 5 values: {ema_values.tail(5).values}")

# Relative Strength Index
print("\n3. Relative Strength Index (14-period)")
rsi_values = rsi(prices, 14)
print(f"Last RSI value: {rsi_values.iloc[-1]:.2f}")
if rsi_values.iloc[-1] > 70:
    print("   -> Overbought condition")
elif rsi_values.iloc[-1] < 30:
    print("   -> Oversold condition")
else:
    print("   -> Neutral condition")

# MACD
print("\n4. MACD")
macd_line, signal_line, histogram = macd(prices, 12, 26, 9)
print(f"Last MACD Line: {macd_line.iloc[-1]:.4f}")
print(f"Last Signal Line: {signal_line.iloc[-1]:.4f}")
print(f"Last Histogram: {histogram.iloc[-1]:.4f}")

# Bollinger Bands
print("\n5. Bollinger Bands (20-period, 2 std dev)")
upper, middle, lower = bollinger_bands(prices, 20, 2.0)
print(f"Last Upper Band: {upper.iloc[-1]:.2f}")
print(f"Last Middle Band: {middle.iloc[-1]:.2f}")
print(f"Last Lower Band: {lower.iloc[-1]:.2f}")

# Stochastic Oscillator
print("\n6. Stochastic Oscillator")
high = pd.Series([p + 2 for p in prices])  # Simulated high prices
low = pd.Series([p - 2 for p in prices])   # Simulated low prices
k, d = stochastic_oscillator(high, low, prices, 14, 3, 3)
print(f"Last %K: {k.iloc[-1]:.2f}")
print(f"Last %D: {d.iloc[-1]:.2f}")

print("\n" + "=" * 60)
print("All indicators calculated successfully!")
print("=" * 60)
