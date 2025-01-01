#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

Demonstration with improved reward logic:
  - Min hold bars to reduce flips
  - Reduced transaction/flipping costs
  - Clamped rewards ±10
  - Early stop if account_balance < 50%
  - Partial logs with tabulate
"""

import os
import numpy as np
import pandas as pd

# Local imports
from utils import configure_gpu, partial_fit_scaler, load_es_data
from technical_indicators import compute_indicators_parallel
from bayes import calculate_bayes_prob
from agent import TradingAgent

# Attempt tabulate for partial logs
try:
    from tabulate import tabulate
    USE_TABULATE = True
except ImportError:
    USE_TABULATE = False

def run_training():
    # GPU config
    configure_gpu()

    # 1) Load data
    data_file_path = "ES_1min.txt"  # Adjust if needed
    try:
        raw_data = load_es_data(data_file_path)
        print("Loaded ES data:")
        print(raw_data.head())
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    # 2) Add day-of-week & day-of-year
    raw_data["DayOfWeek"] = raw_data.index.dayofweek.astype(float)  # 0=Mon..6=Sun
    raw_data["DayOfYear"] = raw_data.index.dayofyear.astype(float)  # 1..366

    # 3) Compute indicators
    es_data = compute_indicators_parallel(raw_data, chunk_size=20000, overlap=30, n_jobs=-1)

    # 4) Bayesian prob
    es_data["Bayes_Prob"] = calculate_bayes_prob(es_data, lookback=100, n_jobs=-1)
    es_data["Bayes_Prob"] = np.clip(es_data["Bayes_Prob"], 0.0, 1.0)

    # Fill missing RSI, MACD, etc.
    for col, default_val in [
        ("RSI", 50.0),
        ("MACD_Line", 0.0),
        ("MACD_Signal", 0.0),
    ]:
        if col not in es_data.columns:
            es_data[col] = default_val
        es_data[col] = es_data[col].fillna(default_val, inplace=True)

    # Add placeholders for logging
    es_data["Action"] = "Hold"
    es_data["Profit/Loss"] = 0.0
    es_data["Position Quantity"] = 0

    # 5) Partial-fit scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    es_data = partial_fit_scaler(es_data, scaler, chunk_size=10000)

    # Identify numeric columns
    numeric_cols = es_data.select_dtypes(include=["number"]).columns
    if es_data[numeric_cols].isna().any().any():
        raise ValueError("Found NaNs after scaling or computations!")

    # 6) Prepare RL environment
    excluded_cols = ["Profit/Loss", "Position Quantity", "Action"]
    numeric_state_cols = [c for c in numeric_cols if c not in excluded_cols]
    state_dim = len(numeric_state_cols)
    action_dim = 4  # (0=GoLong,1=Hold,2=Close,3=GoShort)

    # Build the agent
    agent = TradingAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        gamma=0.95,
        buffer_size=50000
    )
    print("Initialized trading agent.")

    # 7) Reward & session settings
    ALPHA = 1.0              # smaller alpha
    TRADE_OPEN_COST = 0.5    # smaller open cost
    FLIP_PENALTY = 0.5       # smaller flip penalty
    TRANSACTION_COST = 0.5   # smaller transaction
    REWARD_CLAMP = 10.0      # more aggressive clamp
    MIN_HOLD_BARS = 5        # must hold position at least 5 bars
    EARLY_STOP_THRESHOLD = 0.5  # 50% of initial capital

    num_episodes = 5
    step_records = []

    for episode in range(num_episodes):
        print(f"\n=== EPISODE {episode+1}/{num_episodes} ===")
        epsilon = max(0.1, 1.0 - 0.1 * episode)
        print(f"  Epsilon = {epsilon}")

        # Reset environment
        account_balance = 100000.0
        position_dir = None  # "long"/"short"/None
        positions = 0
        total_profit = 0.0
        hold_counter = 0  # how many bars we've held current position

        # Loop over data
        for i in range(len(es_data) - 1):
            row = es_data.iloc[i]
            next_row = es_data.iloc[i + 1]

            # Build state
            state = row[numeric_state_cols].to_numpy(dtype=np.float32)
            next_state = next_row[numeric_state_cols].to_numpy(dtype=np.float32)

            bayes_prob = float(row["Bayes_Prob"])
            action = agent.act(state, epsilon)
            old_pos = position_dir
            reward = 0.0

            # We'll define flipping_action
            flipping_action = False
            if position_dir == "long" and action == 3:
                flipping_action = True
            elif position_dir == "short" and action == 0:
                flipping_action = True

            # If we have a position, increment hold_counter each bar
            if position_dir is not None:
                hold_counter += 1

            # If flipping action is True but we haven't held long enough:
            if flipping_action and hold_counter < MIN_HOLD_BARS:
                # penalize flipping or override action
                reward -= 5.0
                # override flipping to action=1 (Hold)
                action = 1

            # Now handle final chosen action
            if action == 0:  # Go Long
                if position_dir != "long":
                    reward += ALPHA * (bayes_prob - 0.5)
                    reward -= (TRADE_OPEN_COST + TRANSACTION_COST)
                    if position_dir == "short":
                        reward -= FLIP_PENALTY
                    position_dir = "long"
                    positions = 1
                    hold_counter = 0
                    es_data.loc[es_data.index[i], "Action"] = "Go Long"
                    es_data.loc[es_data.index[i], "Position Quantity"] = 1

            elif action == 3:  # Go Short
                if position_dir != "short":
                    reward += ALPHA * (0.5 - bayes_prob)
                    reward -= (TRADE_OPEN_COST + TRANSACTION_COST)
                    if position_dir == "long":
                        reward -= FLIP_PENALTY
                    position_dir = "short"
                    positions = -1
                    hold_counter = 0
                    es_data.loc[es_data.index[i], "Action"] = "Go Short"
                    es_data.loc[es_data.index[i], "Position Quantity"] = -1

            elif action == 1:  # Hold
                if position_dir == "long" and bayes_prob > 0.5:
                    reward += 0.5
                elif position_dir == "short" and bayes_prob < 0.5:
                    reward += 0.5
                es_data.loc[es_data.index[i], "Action"] = "Hold"

            elif action == 2:  # Close
                if positions != 0:
                    reward -= TRANSACTION_COST
                else:
                    reward -= 1.0
                position_dir = None
                positions = 0
                hold_counter = 0
                es_data.loc[es_data.index[i], "Action"] = "Close"
                es_data.loc[es_data.index[i], "Position Quantity"] = 0

            # clamp
            if reward > REWARD_CLAMP:
                reward = REWARD_CLAMP
            elif reward < -REWARD_CLAMP:
                reward = -REWARD_CLAMP

            total_profit += reward
            account_balance += reward

            done = (i == len(es_data) - 2)
            agent.memory.store(state, action, reward, next_state, done)
            agent.train(batch_size=64, print_freq=200)

            step_info = {
                "Episode": episode + 1,
                "Step": i,
                "OldPos": old_pos,
                "Action": action,
                "NewPos": position_dir,
                "Reward": round(reward, 4),
                "TotalProfit": round(total_profit, 4),
                "Balance": round(account_balance, 4),
                "BayesProb": round(bayes_prob, 4),
            }
            step_records.append(step_info)

            if (i > 0) and (i % 200 == 0):
                print(f"  [Ep {episode+1}, Step {i}] Action={action}, OldPos={old_pos}->NewPos={position_dir}, "
                      f"Reward={reward:.2f}, Profit={total_profit:.2f}, Bal={account_balance:.2f}")
                if USE_TABULATE:
                    partial_df = pd.DataFrame(step_records).tail(5)
                    print(tabulate(partial_df, headers="keys", tablefmt="psql"))
                else:
                    print(pd.DataFrame(step_records).tail(5))

            # Early stop if balance < threshold
            if account_balance < 100000.0 * EARLY_STOP_THRESHOLD:
                print(f"  Early stop triggered. Bal={account_balance:.2f} < {EARLY_STOP_THRESHOLD*100:.2f}% of initial.")
                break

        print(f"Episode {episode+1} done. Profit={total_profit:.2f}, Final Bal={account_balance:.2f}")
        agent.save_model("trading_model.h5")

    # Save final data with actions
    output_file_path = "es_data_with_actions.csv"
    es_data.to_csv(output_file_path)
    print(f"\nFinal data with actions saved to {output_file_path}")

    # Save step records
    step_df = pd.DataFrame(step_records)
    step_df.to_csv("training_step_records.csv", index=False)
    print("Step-by-step records saved to training_step_records.csv")

    # Reload model
    agent.load_model("trading_model.h5")
    print("Model reloaded. Done.")


if __name__ == "__main__":
    run_training()
