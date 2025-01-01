#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_code.py

Demonstration of:
  - Loading data from a .txt file with columns [timestamp, open, high, low, close, volume]
  - Using parallel technical indicators (technical_indicators.py)
  - Using a Bayesian probability approach in parallel (bayes_joblib.py)
  - Partial-fitting a scaler with df.loc[...] to avoid chain assignments
  - Reinforcement Learning (RL) logic with a DQN agent
  - Refinements to reward structure: PnL-based rewards, holding rewards, flipping penalties, and clamping.
  - Additional data-cleaning steps to prevent 'infinity' or dtype errors
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Import your parallel computations
from technical_indicators import compute_indicators_parallel
from bayes_joblib import calculate_bayes_prob

# Optional: tabulate for nicer table printing
try:
    from tabulate import tabulate
    USE_TABULATE = True
except ImportError:
    USE_TABULATE = False

###############################################################################
# 1) GPU CONFIG (optional)
###############################################################################
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.optimizer.set_jit(True)  # Optional: XLA acceleration
        except RuntimeError as e:
            print("GPU config error:", e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
configure_gpu()

###############################################################################
# 2) DATA LOADING FUNCTION FOR `.txt` FILES
###############################################################################
def load_es_data(file_path):
    """
    Load ES data from a `.txt` file with columns:
      ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    Delimiter assumed ',' unless you change 'delimiter=' param.
    """
    try:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.read_csv(file_path, names=columns, parse_dates=['timestamp'], delimiter=',')
        if 'timestamp' not in df.columns:
            raise ValueError("Expected 'timestamp' column in the input file.")
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError as e:
        raise ValueError(f"Error loading file: {file_path}. Details: {e}")

###############################################################################
# 3) PARTIAL-FIT SCALER WITH CLEANING
###############################################################################
def partial_fit_scaler(df, scaler, chunk_size=10000):
    """
    1) Convert numeric columns to float64
    2) Replace inf or large values with NaN
    3) partial_fit in chunks
    4) transform in chunks
    5) Cast scaled data to float64

    Prevents "incompatible dtype" warnings and "value too large for float64" errors.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=['number']).columns

    # Ensure columns are float64 from the start
    df_copy[numeric_cols] = df_copy[numeric_cols].astype(np.float64, errors='ignore')

    # Replace infinite values with NaN
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill or drop NaNs (here we fill with 0.0)
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0.0)

    n = len(df_copy)

    # partial_fit in chunks
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        row_slice = df_copy.index[start:end]
        chunk = df_copy.loc[row_slice, numeric_cols]

        # double-check no leftover infinities
        if not np.isfinite(chunk.values).all():
            raise ValueError(f"Found non-finite values in chunk rows {start} to {end}. Investigate data.")

        scaler.partial_fit(chunk)
        start = end

    # transform in chunks
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        row_slice = df_copy.index[start:end]
        chunk = df_copy.loc[row_slice, numeric_cols]

        scaled_vals = scaler.transform(chunk)
        scaled_vals = scaled_vals.astype(np.float64, copy=False)

        df_copy.loc[row_slice, numeric_cols] = scaled_vals
        start = end

    return df_copy

###############################################################################
# 4) REPLAY BUFFER & TRADING AGENT (DQN)
###############################################################################
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim):
        self.states = np.zeros((buffer_size, state_dim, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim, 1), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size

    def store(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state.reshape(-1, 1)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state.reshape(-1, 1)
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

class TradingAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.memory = ReplayBuffer(50000, state_dim)
        self.model = self.build_model(lr)
        self.train_step_counter = 0

    def build_model(self, lr):
        inputs = Input(shape=(self.state_dim, 1))
        x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=1)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(self.action_dim, activation='linear')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
        return model

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = np.expand_dims(state, axis=(0, 2))  # shape => (1, state_dim, 1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size=64, print_freq=200):
        if self.memory.size < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_batch(batch_size)

        target_qs = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            q_update = rewards[i]
            if not dones[i]:
                q_update += self.gamma * np.max(next_qs[i])
            target_qs[i, actions[i]] = q_update

        history = self.model.fit(states, target_qs, epochs=1, verbose=0, batch_size=batch_size)
        self.train_step_counter += 1
        if self.train_step_counter % print_freq == 0:
            loss_val = history.history['loss'][0]
            print(f"[Train Step {self.train_step_counter}] - DQN Loss: {loss_val:.4f}")

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)

###############################################################################
# 5) MAIN SCRIPT
###############################################################################
if __name__ == '__main__':
    data_file_path = '/mnt/c/Users/alial/PycharmProjects/ES_Reinforcement_learning/ES_1min.txt'
    model_save_path = '/mnt/c/Users/alial/PycharmProjects/ES_Reinforcement_learning/trading_model.h5'
    output_file_path = '/mnt/c/Users/alial/PycharmProjects/ES_Reinforcement_learning/es_data_with_actions.csv'

    # 1) Load raw data
    try:
        es_data = load_es_data(data_file_path)
        print("Loaded ES data:")
        print(es_data.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 2) Compute parallel indicators
    es_data = compute_indicators_parallel(es_data, chunk_size=20000, overlap=30, n_jobs=-1)

    # Replace or fill infinite values early
    es_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    es_data.fillna(0.0, inplace=True)

    # 3) Compute parallel Bayesian probability
    es_data['Bayes_Prob'] = calculate_bayes_prob(es_data, lookback=100, n_jobs=-1)

    # Again replace infinite or NaN after bayes
    es_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    es_data.fillna(0.0, inplace=True)

    # Add placeholders
    es_data['Action'] = 'Hold'
    es_data['Profit/Loss'] = 0.0
    es_data['Position Quantity'] = 0

    # 4) PARTIAL-FIT SCALER (with data cleaning)
    scaler = StandardScaler()
    try:
        es_data = partial_fit_scaler(es_data, scaler, chunk_size=10000)
    except ValueError as err:
        print(f"Scaling error: {err}")
        exit(1)

    # 5) RL environment
    excluded_cols = ['Profit/Loss', 'Position Quantity', 'Action']
    numeric_cols = [c for c in es_data.select_dtypes(include=['number']).columns if c not in excluded_cols]
    state_dim = len(numeric_cols)
    action_dim = 4  # 0=GoLong, 1=Hold, 2=Close, 3=GoShort

    agent = TradingAgent(state_dim=state_dim, action_dim=action_dim, lr=0.0005, gamma=0.95)
    print("Initialized trading agent.")

    # Example training hyperparameters
    num_episodes = 5
    ALPHA = 10.0
    TRADE_OPEN_COST = 2.0
    FLIP_PENALTY = 2.0
    TRANSACTION_COST = 2.50
    REWARD_CLAMP = 50.0
    HOLD_REWARD_SCALE = 0.1

    account_balance = 100000.0
    position_dir = None
    positions = 0

    step_records = []
    for episode in range(num_episodes):
        print(f"\n=== EPISODE {episode+1}/{num_episodes} ===")
        epsilon = max(0.1, 1.0 - 0.1 * episode)
        total_profit = 0.0

        for i in range(len(es_data) - 1):
            row = es_data.iloc[i]
            next_row = es_data.iloc[i + 1]

            # Build states as float32
            state = row[numeric_cols].to_numpy(dtype=np.float32)
            next_state = next_row[numeric_cols].to_numpy(dtype=np.float32)

            current_close = float(row['close'])
            next_close = float(next_row['close'])
            bayes_prob = float(row['Bayes_Prob'])

            action = agent.act(state, epsilon)

            reward = 0.0
            old_pos = position_dir

            # ---------------------------
            # Reward Logic (PnL, holding, flipping, etc.)
            # ---------------------------
            if action == 0:  # Go Long
                if position_dir != "long":
                    # Bayesian-based reward
                    reward += ALPHA * (bayes_prob - 0.5)
                    # Cost to open a trade
                    reward -= (TRADE_OPEN_COST + TRANSACTION_COST)
                    # Penalty if flipping from short
                    if position_dir == "short":
                        reward -= FLIP_PENALTY
                    position_dir = "long"
                    positions = 1
                    idx_label = es_data.index[i]
                    es_data.loc[idx_label, 'Action'] = "Go Long"
                    es_data.loc[idx_label, 'Position Quantity'] = 1

            elif action == 3:  # Go Short
                if position_dir != "short":
                    # Bayesian-based reward
                    reward += ALPHA * (0.5 - bayes_prob)
                    # Cost to open
                    reward -= (TRADE_OPEN_COST + TRANSACTION_COST)
                    # Penalty if flipping from long
                    if position_dir == "long":
                        reward -= FLIP_PENALTY
                    position_dir = "short"
                    positions = -1
                    idx_label = es_data.index[i]
                    es_data.loc[idx_label, 'Action'] = "Go Short"
                    es_data.loc[idx_label, 'Position Quantity'] = -1

            elif action == 1:  # Hold
                # If we hold a position, small reward if in the right direction
                if position_dir == "long" and next_close > current_close:
                    reward += HOLD_REWARD_SCALE * (next_close - current_close)
                elif position_dir == "short" and next_close < current_close:
                    reward += HOLD_REWARD_SCALE * (current_close - next_close)
                idx_label = es_data.index[i]
                es_data.loc[idx_label, 'Action'] = "Hold"

            elif action == 2:  # Close
                if positions != 0:
                    # Calculate real PnL
                    pnl = (next_close - current_close) * positions
                    reward += pnl - TRANSACTION_COST
                    total_profit += pnl
                else:
                    # Penalty for trying to close with no position
                    reward -= 1.0
                position_dir = None
                positions = 0
                idx_label = es_data.index[i]
                es_data.loc[idx_label, 'Action'] = "Close"
                es_data.loc[idx_label, 'Position Quantity'] = 0

            # clamp reward
            reward = max(-REWARD_CLAMP, min(REWARD_CLAMP, reward))
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

            # Periodic logging
            if i > 0 and i % 200 == 0:
                print(f"  [Ep {episode+1}, Step {i}] Act={action}, OldPos={old_pos}->NewPos={position_dir}, "
                      f"Reward={reward:.2f}, Profit={total_profit:.2f}, Bal={account_balance:.2f}")
                if USE_TABULATE:
                    partial_df = pd.DataFrame(step_records).tail(5)
                    print(tabulate(partial_df, headers='keys', tablefmt='psql'))
                else:
                    print(pd.DataFrame(step_records).tail(5))

        print(f"Episode {episode+1} done. Profit={total_profit:.2f}, Balance={account_balance:.2f}")
        agent.save_model(model_save_path)

    # Save final data
    es_data.to_csv(output_file_path)
    print(f"\nFinal data with actions saved to: {output_file_path}")

    # Save step records
    step_record_path = '/mnt/c/Users/alial/PycharmProjects/ES_Reinforcement_learning/training_step_records.csv'
    step_df = pd.DataFrame(step_records)
    step_df.to_csv(step_record_path, index=False)
    print(f"Step-by-step records saved to: {step_record_path}")

    # Reload model
    agent.load_model(model_save_path)
    print("Model reloaded. Done.")
