"""
Technical Indicators Implementation

This module contains implementations of common technical indicators
used in trading analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def sma(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: Price data as pandas Series or numpy array
        period: Number of periods for the moving average
        
    Returns:
        SMA values as the same type as input
        
    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14, 15])
        >>> sma(prices, 3)
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=period).mean()
    else:
        return pd.Series(data).rolling(window=period).mean().values


def ema(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: Price data as pandas Series or numpy array
        period: Number of periods for the moving average
        
    Returns:
        EMA values as the same type as input
        
    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14, 15])
        >>> ema(prices, 3)
    """
    if isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean()
    else:
        return pd.Series(data).ewm(span=period, adjust=False).mean().values


def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: Price data as pandas Series or numpy array
        period: Number of periods for RSI calculation (default: 14)
        
    Returns:
        RSI values (0-100) as the same type as input
        
    Example:
        >>> prices = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83])
        >>> rsi(prices, 14)
    """
    return_array = not isinstance(data, pd.Series)
    
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    if return_array:
        return rsi_values.values
    else:
        return rsi_values


def macd(data: Union[pd.Series, np.ndarray], 
         fast_period: int = 12, 
         slow_period: int = 26, 
         signal_period: int = 9) -> Tuple[Union[pd.Series, np.ndarray], 
                                           Union[pd.Series, np.ndarray], 
                                           Union[pd.Series, np.ndarray]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price data as pandas Series or numpy array
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
        
    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        >>> macd_line, signal_line, histogram = macd(prices)
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
        return_array = True
    else:
        return_array = False
    
    # Calculate EMAs
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    if return_array:
        return macd_line.values, signal_line.values, histogram.values
    else:
        return macd_line, signal_line, histogram


def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                   period: int = 20, 
                   std_dev: float = 2.0) -> Tuple[Union[pd.Series, np.ndarray], 
                                                   Union[pd.Series, np.ndarray], 
                                                   Union[pd.Series, np.ndarray]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price data as pandas Series or numpy array
        period: Number of periods for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
        
    Example:
        >>> prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        >>> upper, middle, lower = bollinger_bands(prices)
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
        return_array = True
    else:
        return_array = False
    
    # Calculate middle band (SMA)
    middle_band = data.rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = data.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    if return_array:
        return upper_band.values, middle_band.values, lower_band.values
    else:
        return upper_band, middle_band, lower_band


def stochastic_oscillator(high: Union[pd.Series, np.ndarray],
                          low: Union[pd.Series, np.ndarray],
                          close: Union[pd.Series, np.ndarray],
                          period: int = 14,
                          smooth_k: int = 3,
                          smooth_d: int = 3) -> Tuple[Union[pd.Series, np.ndarray], 
                                                      Union[pd.Series, np.ndarray]]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High prices as pandas Series or numpy array
        low: Low prices as pandas Series or numpy array
        close: Close prices as pandas Series or numpy array
        period: Lookback period (default: 14)
        smooth_k: Smoothing period for %K (default: 3)
        smooth_d: Smoothing period for %D (default: 3)
        
    Returns:
        Tuple of (%K, %D)
        
    Example:
        >>> high = pd.Series([15, 16, 17, 18, 19, 20])
        >>> low = pd.Series([10, 11, 12, 13, 14, 15])
        >>> close = pd.Series([12, 13, 14, 15, 16, 17])
        >>> k, d = stochastic_oscillator(high, low, close)
    """
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        return_array = True
    else:
        return_array = False
    
    # Calculate lowest low and highest high over the period
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    # Calculate %K
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Smooth %K
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    
    # Calculate %D (signal line)
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    
    if return_array:
        return k_smooth.values, d_smooth.values
    else:
        return k_smooth, d_smooth
