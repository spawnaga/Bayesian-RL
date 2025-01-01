# -*- coding: utf-8 -*-
"""
Created on Mon December 30, 2024

@author: Alex Oraibi

Description:
  Parallel computations of naive Bayesian probabilities using joblib.
  Example usage in main_code.py:
     from bayes_joblib import calculate_bayes_prob
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def compute_prob_for_index(i, bar_up, bar_up_next, lookback):
    """
    For a single index i, compute:
      P(next bar up | bar_up[i]) by looking back 'lookback' bars
      and counting how often bar_up = bar_up[i], among those, how many next_up=1.
    """
    n = len(bar_up)
    if i >= n - 1:
        return 0.5  # no next bar => fallback

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

    mask = (subset_bar == curr_state) & (subset_next == 1)
    count_e_up = mask.sum()
    prob = count_e_up / count_e
    return prob

def calculate_bayes_prob(df, lookback=100, n_jobs=-1):
    """
    Parallel version of naive Bayesian probability.
    bar_up[i] = 1 if close[i]>open[i], else 0
    bar_up_next = shifted version for next bar.

    Returns a Series 'Bayes_Prob'.
    """
    # define bar_up as (close>open)
    bar_up = (df['close'] > df['open']).astype(int).values
    # next bar up => shift by -1
    bar_up_next = np.roll(bar_up, -1)
    bar_up_next[-1] = 0

    n = len(df)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_prob_for_index)(i, bar_up, bar_up_next, lookback)
        for i in range(n)
    )

    return pd.Series(results, index=df.index, name='Bayes_Prob')
