"""
Unit tests for technical indicators.
"""

import numpy as np
import pandas as pd
import pytest
from trading_indicators import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    stochastic_oscillator,
)


class TestSMA:
    """Tests for Simple Moving Average."""
    
    def test_sma_with_series(self):
        """Test SMA calculation with pandas Series."""
        data = pd.Series([10, 11, 12, 13, 14, 15])
        result = sma(data, 3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 11.0  # (10+11+12)/3
        assert result.iloc[3] == 12.0  # (11+12+13)/3
    
    def test_sma_with_array(self):
        """Test SMA calculation with numpy array."""
        data = np.array([10, 11, 12, 13, 14, 15])
        result = sma(data, 3)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 11.0


class TestEMA:
    """Tests for Exponential Moving Average."""
    
    def test_ema_with_series(self):
        """Test EMA calculation with pandas Series."""
        data = pd.Series([10, 11, 12, 13, 14, 15])
        result = ema(data, 3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.iloc[0] == 10.0  # First value equals first data point
        assert result.iloc[-1] > result.iloc[0]  # Should be trending up
    
    def test_ema_with_array(self):
        """Test EMA calculation with numpy array."""
        data = np.array([10, 11, 12, 13, 14, 15])
        result = ema(data, 3)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)


class TestRSI:
    """Tests for Relative Strength Index."""
    
    def test_rsi_with_series(self):
        """Test RSI calculation with pandas Series."""
        # Create data with clear trend
        data = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                         45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
        result = rsi(data, 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # RSI should be between 0 and 100
        assert all((result.dropna() >= 0) & (result.dropna() <= 100))
    
    def test_rsi_with_array(self):
        """Test RSI calculation with numpy array."""
        data = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
        result = rsi(data, 14)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_rsi_overbought_oversold(self):
        """Test RSI identifies overbought/oversold conditions."""
        # Create strongly upward trending data
        uptrend = pd.Series(range(1, 31))
        result_up = rsi(uptrend, 14)
        
        # RSI should be high for uptrend
        assert result_up.iloc[-1] > 70
        
        # Create strongly downward trending data
        downtrend = pd.Series(range(30, 0, -1))
        result_down = rsi(downtrend, 14)
        
        # RSI should be low for downtrend
        assert result_down.iloc[-1] < 30


class TestMACD:
    """Tests for MACD."""
    
    def test_macd_with_series(self):
        """Test MACD calculation with pandas Series."""
        data = pd.Series(range(1, 51))
        macd_line, signal_line, histogram = macd(data)
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd_line) == len(data)
    
    def test_macd_with_array(self):
        """Test MACD calculation with numpy array."""
        data = np.array(range(1, 51))
        macd_line, signal_line, histogram = macd(data)
        
        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        assert len(macd_line) == len(data)
    
    def test_macd_histogram(self):
        """Test MACD histogram calculation."""
        data = pd.Series(range(1, 51))
        macd_line, signal_line, histogram = macd(data)
        
        # Histogram should equal MACD - Signal
        assert np.allclose(histogram, macd_line - signal_line, equal_nan=True)


class TestBollingerBands:
    """Tests for Bollinger Bands."""
    
    def test_bollinger_bands_with_series(self):
        """Test Bollinger Bands calculation with pandas Series."""
        data = pd.Series(range(1, 51))
        upper, middle, lower = bollinger_bands(data, 20, 2.0)
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(data)
    
    def test_bollinger_bands_with_array(self):
        """Test Bollinger Bands calculation with numpy array."""
        data = np.array(range(1, 51))
        upper, middle, lower = bollinger_bands(data, 20, 2.0)
        
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert len(upper) == len(data)
    
    def test_bollinger_bands_order(self):
        """Test that upper > middle > lower."""
        data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        upper, middle, lower = bollinger_bands(data, 20, 2.0)
        
        # Check order (ignoring NaN values)
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        assert all(upper[valid_indices] >= middle[valid_indices])
        assert all(middle[valid_indices] >= lower[valid_indices])


class TestStochasticOscillator:
    """Tests for Stochastic Oscillator."""
    
    def test_stochastic_with_series(self):
        """Test Stochastic Oscillator with pandas Series."""
        high = pd.Series(range(15, 35))
        low = pd.Series(range(10, 30))
        close = pd.Series(range(12, 32))
        
        k, d = stochastic_oscillator(high, low, close, 14, 3, 3)
        
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert len(k) == len(high)
        assert len(d) == len(high)
    
    def test_stochastic_with_array(self):
        """Test Stochastic Oscillator with numpy arrays."""
        high = np.array(range(15, 35))
        low = np.array(range(10, 30))
        close = np.array(range(12, 32))
        
        k, d = stochastic_oscillator(high, low, close, 14, 3, 3)
        
        assert isinstance(k, np.ndarray)
        assert isinstance(d, np.ndarray)
        assert len(k) == len(high)
    
    def test_stochastic_range(self):
        """Test that Stochastic values are between 0 and 100."""
        high = pd.Series([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 33, 34])
        low = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29])
        close = pd.Series([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                          23, 24, 25, 26, 27, 28, 29, 30, 31])
        
        k, d = stochastic_oscillator(high, low, close, 14, 3, 3)
        
        # Values should be between 0 and 100
        assert all((k.dropna() >= 0) & (k.dropna() <= 100))
        assert all((d.dropna() >= 0) & (d.dropna() <= 100))


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_data(self):
        """Test with empty data."""
        data = pd.Series([])
        result = sma(data, 3)
        assert len(result) == 0
    
    def test_period_larger_than_data(self):
        """Test when period is larger than data length."""
        data = pd.Series([1, 2, 3])
        result = sma(data, 10)
        assert all(result.isna())
    
    def test_single_value(self):
        """Test with single value."""
        data = pd.Series([10])
        result = ema(data, 1)
        assert result.iloc[0] == 10.0
