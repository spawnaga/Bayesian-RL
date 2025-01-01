#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

Houses utility functions:
  1) GPU config (configure_gpu)
  2) partial_fit_scaler (unchanged except minor improvements)
  3) (Optional) load_es_data function (if needed here instead of main)
  4) Additional small utilities (like reward clamping, etc.) if desired
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def configure_gpu():
    """
    Attempt to configure GPU memory growth & optionally XLA for speed.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Optional XLA
            tf.config.optimizer.set_jit(True)
        except RuntimeError as e:
            print(f"GPU config error: {e}")

def partial_fit_scaler(df: pd.DataFrame, scaler: StandardScaler, chunk_size=10000) -> pd.DataFrame:
    """
    Partial-fit and transform data in chunks to avoid memory issues,
    also to avoid SettingWithCopy warnings.

    Steps:
      1) Convert numeric columns to float64
      2) Replace inf or large values with NaN
      3) partial_fit in chunks
      4) transform in chunks
      5) cast scaled data to float64
    """

    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=['number']).columns

    # 1) Ensure columns are float64
    df_copy[numeric_cols] = df_copy[numeric_cols].astype(np.float64, errors='ignore')

    # 2) Replace infinite values with NaN
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2b) Fill or drop NaNs
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0.0)

    n = len(df_copy)
    # partial_fit in chunks
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        row_slice = df_copy.index[start:end]
        chunk = df_copy.loc[row_slice, numeric_cols]

        if not np.isfinite(chunk.values).all():
            raise ValueError(f"Non-finite values in chunk {start}:{end}. Investigate data.")

        scaler.partial_fit(chunk)
        start = end

    # transform in chunks
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        row_slice = df_copy.index[start:end]
        chunk = df_copy.loc[row_slice, numeric_cols]

        scaled_vals = scaler.transform(chunk)
        # cast scaled vals to float64
        scaled_vals = scaled_vals.astype(np.float64, copy=False)
        df_copy.loc[row_slice, numeric_cols] = scaled_vals
        start = end

    return df_copy

# If you prefer to keep the load_es_data here, you can place it as well:
def load_es_data(file_path: str) -> pd.DataFrame:
    """
    Example function to load a .txt file with:
      [timestamp, open, high, low, close, volume]
    """
    try:
        columns = ['timestamp','open','high','low','close','volume']
        df = pd.read_csv(file_path, names=columns, parse_dates=['timestamp'], delimiter=',')
        if 'timestamp' not in df.columns:
            raise ValueError("Expected 'timestamp' column in file.")
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError as e:
        raise ValueError(f"Error loading file: {file_path}. Details: {e}")
