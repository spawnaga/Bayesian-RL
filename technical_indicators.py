#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
technical_indicators.py

Provides:
  1) RSI (relative_strength_index)
  2) MACD (macd_line, signal_line)
  3) Bollinger Bands (middle, upper, lower)
  4) Parallel computation helpers
"""

import pandas as pd
from joblib import Parallel, delayed

def rsi(series: pd.Series, period=14) -> pd.Series:
    """
    Compute the RSI for a price series using a rolling window approach.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, short_window=12, long_window=26, signal_window=9):
    """
    Compute MACD line (ema_short - ema_long) and signal line (ema of macd_line).
    """
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(series: pd.Series, period=20, num_std=2):
    """
    Compute Bollinger Bands (center=ma, upper, lower).
    """
    sma = series.rolling(window=period).mean()
    stddev = series.rolling(window=period).std()
    upper = sma + (num_std * stddev)
    lower = sma - (num_std * stddev)
    return sma, upper, lower

def _compute_indicators_for_chunk(data_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Helper for computing RSI, MACD, Bollinger on a data chunk.
    """
    chunk = data_chunk.copy()
    close_series = chunk['close']

    # RSI
    chunk['RSI'] = rsi(close_series)

    # MACD
    macd_line, signal_line = macd(close_series)
    chunk['MACD_Line'] = macd_line
    chunk['MACD_Signal'] = signal_line

    # Bollinger
    sma, upper, lower = bollinger_bands(close_series)
    chunk['Bollinger_MA'] = sma
    chunk['Bollinger_Upper'] = upper
    chunk['Bollinger_Lower'] = lower

    return chunk

def _indicator_chunk_with_overlap(chunk_df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Called by the parallel loop. Just calls `_compute_indicators_for_chunk`.
    """
    return _compute_indicators_for_chunk(chunk_df)

def compute_indicators_parallel(df: pd.DataFrame, chunk_size=20000, overlap=30, n_jobs=-1) -> pd.DataFrame:
    """
    Parallel computation for large DataFrames:
      1) Split into chunks with slight overlap
      2) Compute indicators in parallel
      3) Recombine

    The overlap ensures rolling windows near chunk boundaries aren't truncated.
    """
    df_sorted = df.sort_index()
    n = len(df_sorted)
    starts = range(0, n, chunk_size)
    tasks = []

    for start in starts:
        end = min(start + chunk_size, n)
        ext_end = min(end + overlap, n)
        subset = df_sorted.iloc[start:ext_end].copy()
        tasks.append((subset, start, end))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_indicator_chunk_with_overlap)(subset, start, end)
        for (subset, start, end) in tasks
    )

    combined = pd.concat(results)
    # groupby(level=0) in case index duplicates from overlapping chunks
    combined = combined.sort_index().groupby(level=0).first()
    return combined
