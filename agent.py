#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agent.py

Houses:
  1) ReplayBuffer class
  2) TradingAgent (DQN) class with gradient clipping,
     optional smaller reward clamp, or gamma, etc.
"""

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

class ReplayBuffer:
    def __init__(self, buffer_size: int, state_dim: int):
        """
        buffer_size: total capacity
        state_dim: number of features in the state
        """
        self.states = np.zeros((buffer_size, state_dim, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim, 1), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)

        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new transition to the replay buffer.
        state, next_state: shape => (state_dim,)
        """
        idx = self.ptr
        # Reshape for (state_dim, 1) as our DQN expects 3D input
        self.states[idx] = state.reshape(-1, 1)
        self.next_states[idx] = next_state.reshape(-1, 1)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        """
        Randomly sample transitions from the buffer for DQN training.
        """
        if self.size < batch_size:
            # Not enough data in the buffer
            return None
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class TradingAgent:
    def __init__(self, state_dim: int, action_dim: int, lr=1e-4, gamma=0.95, buffer_size=50000):
        """
        state_dim: number of features in state
        action_dim: discrete actions (0=GoLong,1=Hold,2=Close,3=GoShort)
        lr: learning rate
        gamma: discount factor
        buffer_size: replay capacity
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.memory = ReplayBuffer(buffer_size, state_dim)
        self.model = self.build_model(lr)
        self.train_step_counter = 0

    def build_model(self, lr):
        """
        Construct a simple DQN with:
          - Conv1D -> LSTM -> Dense -> BN -> Dense
          - gradient clipping via clipnorm
        """
        inputs = Input(shape=(self.state_dim, 1))
        x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=1)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)

        outputs = Dense(self.action_dim, activation='linear')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=lr, clipnorm=0.5), loss='mse')
        return model

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection.
        state shape => (state_dim,)
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        # Expand dims => (1, state_dim, 1)
        state_reshaped = np.expand_dims(state, axis=(0, 2))
        q_values = self.model.predict(state_reshaped, verbose=0)
        return int(np.argmax(q_values[0]))

    def train(self, batch_size=64, print_freq=200):
        """
        Train DQN on a batch from the replay buffer.
        """
        sampled = self.memory.sample_batch(batch_size)
        if sampled is None:
            return  # Not enough data in buffer

        states, actions, rewards, next_states, dones = sampled

        # Current Q-values
        target_qs = self.model.predict(states, verbose=0)
        # Next Q-values
        next_qs = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            q_update = rewards[i]
            if not dones[i]:
                q_update += self.gamma * np.max(next_qs[i])
            target_qs[i, actions[i]] = q_update

        history = self.model.fit(
            states, target_qs,
            epochs=1, verbose=0, batch_size=batch_size
        )
        self.train_step_counter += 1

        if self.train_step_counter % print_freq == 0:
            loss_val = history.history['loss'][0]
            print(f"[Train Step {self.train_step_counter}] - DQN Loss: {loss_val:.4f}")

    def save_model(self, file_path: str):
        self.model.save(file_path)

    def load_model(self, file_path: str):
        self.model = load_model(file_path)
