# Trading Indicators

A Python library for calculating common technical indicators used in trading and financial analysis.

## Features

This library provides implementations of popular technical indicators:

- **Simple Moving Average (SMA)** - Basic moving average calculation
- **Exponential Moving Average (EMA)** - Weighted moving average giving more weight to recent prices
- **Relative Strength Index (RSI)** - Momentum oscillator measuring speed and magnitude of price changes
- **MACD** - Moving Average Convergence Divergence indicator
- **Bollinger Bands** - Volatility bands placed above and below a moving average
- **Stochastic Oscillator** - Momentum indicator comparing closing price to price range

## Installation

### From source

```bash
git clone https://github.com/amtiawedia/trading-indicators.git
cd trading-indicators
pip install -e .
```

### Dependencies

- Python >= 3.7
- numpy >= 1.21.0
- pandas >= 1.3.0

## Usage

### Simple Moving Average (SMA)

```python
import pandas as pd
from trading_indicators import sma

# With pandas Series
prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
sma_values = sma(prices, period=5)
print(sma_values)
```

### Exponential Moving Average (EMA)

```python
from trading_indicators import ema

prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ema_values = ema(prices, period=5)
print(ema_values)
```

### Relative Strength Index (RSI)

```python
from trading_indicators import rsi

prices = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42])
rsi_values = rsi(prices, period=14)
print(rsi_values)

# RSI > 70 typically indicates overbought
# RSI < 30 typically indicates oversold
```

### MACD (Moving Average Convergence Divergence)

```python
from trading_indicators import macd

prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
macd_line, signal_line, histogram = macd(prices, fast_period=12, slow_period=26, signal_period=9)

print("MACD Line:", macd_line)
print("Signal Line:", signal_line)
print("Histogram:", histogram)
```

### Bollinger Bands

```python
from trading_indicators import bollinger_bands

prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
upper_band, middle_band, lower_band = bollinger_bands(prices, period=20, std_dev=2.0)

print("Upper Band:", upper_band)
print("Middle Band:", middle_band)
print("Lower Band:", lower_band)
```

### Stochastic Oscillator

```python
from trading_indicators import stochastic_oscillator
import pandas as pd

high = pd.Series([15, 16, 17, 18, 19, 20])
low = pd.Series([10, 11, 12, 13, 14, 15])
close = pd.Series([12, 13, 14, 15, 16, 17])

k_percent, d_percent = stochastic_oscillator(high, low, close, period=14, smooth_k=3, smooth_d=3)

print("%K:", k_percent)
print("%D:", d_percent)
```

## Working with NumPy Arrays

All functions accept both pandas Series and numpy arrays:

```python
import numpy as np
from trading_indicators import sma

# With numpy array
prices = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
sma_values = sma(prices, period=5)
print(sma_values)
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=trading_indicators tests/
```

### Project Structure

```
trading-indicators/
├── trading_indicators/
│   ├── __init__.py
│   └── indicators.py
├── tests/
│   ├── __init__.py
│   └── test_indicators.py
├── README.md
├── LICENSE
├── setup.py
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This library implements standard technical indicators commonly used in financial analysis and algorithmic trading.