#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bayes.py

Provides a naive Bayesian probability calculation with minimal improvements:
  - Clipping final probabilities to [0,1].
  - Basic checks to avoid negative or out-of-bound probabilities.

The function:
  calculate_bayes_prob(df, lookback=100, n_jobs=-1)

Data columns used: ['open','close'] => we check bar_up = close>open
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def _compute_prob_for_index(i: int, bar_up: np.ndarray, bar_up_next: np.ndarray, lookback: int) -> float:
    """
    For bar i, compute:
      P(next bar up | bar_up[i]) by looking back 'lookback' bars.
    Returns a float in [0..1].
    """
    n = len(bar_up)
    if i >= n - 1:
        return 0.5  # fallback

    curr_state = bar_up[i]
    start_idx = max(0, i - lookback)
    end_idx = i - 1
    if end_idx < start_idx:
        return 0.5

    subset_bar = bar_up[start_idx:end_idx+1]
    subset_next = bar_up_next[start_idx:end_idx+1]

    count_e = (subset_bar == curr_state).sum()
    if count_e == 0:
        return 0.5

    # among those, how many times was next bar up
    mask = ((subset_bar == curr_state) & (subset_next == 1))
    count_e_up = mask.sum()

    prob = count_e_up / count_e
    # optional: clip strictly
    return float(np.clip(prob, 0.0, 1.0))

def calculate_bayes_prob(df: pd.DataFrame, lookback=100, n_jobs=-1) -> pd.Series:
    """
    bar_up[i] = 1 if close[i]>open[i], else 0
    bar_up_next = shift by -1 => next bar
    compute naive P(next bar up | current bar up/down).
    """
    # define bar_up
    bar_up = (df['close'] > df['open']).astype(int).values
    bar_up_next = np.roll(bar_up, -1)
    bar_up_next[-1] = 0  # last bar fallback

    n = len(df)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_prob_for_index)(i, bar_up, bar_up_next, lookback)
        for i in range(n)
    )

    return pd.Series(results, index=df.index, name='Bayes_Prob')
