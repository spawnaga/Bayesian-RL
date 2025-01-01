# -*- coding: utf-8 -*-
"""
Created on Mon December 30, 2024

@author: Alex Oraibi

Description:
  Parallelized calculation of various technical indicators on large datasets using joblib.
  We'll compute RSI, MACD, Bollinger Bands, etc.

Usage example from main_code.py:
    from technical_indicators import compute_indicators_parallel
    df = compute_indicators_parallel(df, chunk_size=20000, overlap=30, n_jobs=-1)
"""

import pandas as pd
from joblib import Parallel, delayed

def rsi(series, period=14):
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val

def macd(series, short_window=12, long_window=26, signal_window=9):
    """Compute MACD line and signal line."""
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(series, period=20, num_std=2):
    """Compute Bollinger Bands (MA, upper, lower)."""
    sma = series.rolling(window=period).mean()
    stddev = series.rolling(window=period).std()
    upper = sma + (num_std * stddev)
    lower = sma - (num_std * stddev)
    return sma, upper, lower

def compute_indicators_for_chunk(data_chunk):
    """
    Compute multiple indicators for a chunk of data.
    We'll assume we want RSI, MACD, Bollinger on the 'close' price.
    Return the updated DataFrame slice with new columns.
    """
    chunk = data_chunk.copy()  # avoid modifying original reference
    close_series = chunk['close']

    # RSI
    chunk['RSI'] = rsi(close_series)

    # MACD
    macd_line, signal_line = macd(close_series)
    chunk['MACD_Line'] = macd_line
    chunk['MACD_Signal'] = signal_line

    # Bollinger Bands
    sma, upper, lower = bollinger_bands(close_series)
    chunk['Bollinger_MA'] = sma
    chunk['Bollinger_Upper'] = upper
    chunk['Bollinger_Lower'] = lower

    return chunk

def _compute_indicators_chunk_with_overlap(chunk_df, start, end):
    """
    Helper to compute indicators on a chunk with overlap,
    then trim or let the main function reconcile overlaps.
    """
    chunk_df = compute_indicators_for_chunk(chunk_df)
    # We might do more logic here to strictly return only rows in [start, end)
    # if the index is integer-based. If it's datetime, you need a mapping.

    # For simplicity, we just return the entire chunk, letting the main
    # function handle overlap merges with groupby(...).first().
    return chunk_df

def compute_indicators_parallel(df, chunk_size=20000, overlap=30, n_jobs=-1):
    """
    Split the DataFrame into chunks (with overlap) and process each chunk in parallel.
    Then combine.

    'overlap' helps preserve correct rolling calculations near chunk boundaries
    if you rely on rolling windows (e.g., RSI, Bollinger).

    If your index is datetime and not guaranteed contiguous, you might need a
    custom approach to map row numbers to [start:end].
    """
    df_sorted = df.sort_index()
    n = len(df_sorted)
    starts = range(0, n, chunk_size)
    chunks = []

    for start in starts:
        end = min(start + chunk_size, n)
        extended_end = min(end + overlap, n)
        # We copy to avoid SettingWithCopy issues
        chunk_df = df_sorted.iloc[start:extended_end].copy()
        chunks.append((chunk_df, start, end))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_indicators_chunk_with_overlap)(chunk_df, start, end)
        for (chunk_df, start, end) in chunks
    )

    # Combine & merge overlaps
    final_df = pd.concat(results)
    final_df = final_df.sort_index().groupby(level=0).first()  # or .last(), as you see fit
    return final_df
