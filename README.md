# Reinforcement Learning for Trading with Bayesian Signals

## Overview

This project demonstrates a **Reinforcement Learning (RL)** approach to trading futures using a **Deep Q-Network (DQN)**. It integrates:

1. **Technical Indicators**: Parallelized computations (e.g., RSI, MACD, Bollinger Bands) to provide signals for the trading environment.
2. **Bayesian Probabilities**: Parallel computation of **naive Bayesian probabilities** to estimate the likelihood of the next bar being up or down.
3. **Reinforcement Learning**: A DQN agent learns optimal trading strategies based on historical data.

The code is modular and designed to handle **large datasets**, leveraging **parallel processing**, **gradient clipping**, and **partial-fit scaling** to ensure stability and scalability.

---

## Key Features

1. **Parallelized Technical Indicators**
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - These are computed in chunks using `joblib` for efficient parallelization.

2. **Bayesian Probabilities**
   - Computes \( P(\text{next bar up} | \text{current bar state}) \) using a naive Bayesian approach.
   - Uses **parallel processing** to handle large datasets efficiently.

3. **Deep Q-Learning**
   - A DQN agent learns to take actions:
     - `0`: Go Long
     - `1`: Hold
     - `2`: Close
     - `3`: Go Short
   - Actions are based on the agent’s **Q-values**, which are updated during training.

4. **Gradient Clipping**
   - Prevents exploding gradients during DQN training.

5. **Partial-Fit Scaling**
   - Handles large datasets by scaling numeric columns in chunks, avoiding memory issues.

---

## Mathematical Approach

### 1. Technical Indicators

#### (a) **RSI**
\[
\text{RSI} = 100 - \left( \frac{100}{1 + RS} \right), \quad RS = \frac{\text{avg gain}}{\text{avg loss}}
\]
- Identifies overbought/oversold conditions:
  - \( \text{RSI} > 70 \): Overbought (sell signal)
  - \( \text{RSI} < 30 \): Oversold (buy signal)

#### (b) **MACD**
\[
\text{MACD Line} = \text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}}
\]
\[
\text{Signal Line} = \text{EMA}_{\text{MACD}}
\]
- Helps detect momentum shifts and trend reversals.

#### (c) **Bollinger Bands**
\[
\text{Upper Band} = \text{SMA} + (\text{num\_std} \times \text{StdDev})
\]
\[
\text{Lower Band} = \text{SMA} - (\text{num\_std} \times \text{StdDev})
\]
- Identifies price levels relative to a moving average.

### 2. Bayesian Probability

For each row \( i \):
1. Compute:
   - \( \text{bar\_up}[i] = 1 \text{ if close}[i] > \text{open}[i], \text{ else } 0 \)
   - \( \text{bar\_up\_next}[i] = \text{bar\_up}[i+1] \)
2. Within a **lookback window**:
   - Count how often \( \text{bar\_up} \) matches the current bar state.
   - Among those, count how often the next bar was also up.
3. Apply a **naive Bayesian formula**:
\[
P(\text{next bar up} | \text{current bar state}) = \frac{\text{\#(next up | current)}}{\text{\#(current)}}
\]

### 3. Reinforcement Learning with DQN

#### Q-Value Update:
The DQN agent updates its Q-values using:
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a) - Q(s, a) \right]
\]
Where:
- \( s \): Current state (numeric indicators + Bayesian probabilities)
- \( a \): Chosen action
- \( r \): Reward
- \( s' \): Next state
- \( \alpha \): Learning rate
- \( \gamma \): Discount factor

#### Reward Function:
- **Positive Reward**: \( \alpha \times (\text{Bayesian prob} - 0.5) \) for correct direction.
- **Negative Reward**:
  - Transaction costs.
  - Penalty for frequent flips (e.g., long to short).

---

## Coding Approach

### 1. File Structure
- `main_code.py`: Main RL logic and pipeline.
- `technical_indicators.py`: Parallelized computations for RSI, MACD, Bollinger Bands.
- `bayes_joblib.py`: Parallelized Bayesian probability computation.

### 2. GPU Configuration
- Enables **memory growth** and optionally **XLA JIT compilation** for TensorFlow.
- Gradient clipping is applied to ensure stability.

### 3. Key Classes
#### **ReplayBuffer**
Stores states, actions, rewards, and next states for DQN training.

#### **TradingAgent**
Builds and trains the DQN model:
- Uses `Conv1D` and `LSTM` layers to learn temporal patterns.
- Clips gradients using `clipnorm`.

---

## Running the Code

1. **Prepare the Data**
   - Ensure your `.txt` file has the columns:
     ```
     timestamp, open, high, low, close, volume
     ```
   - Example format:
     ```
     2008-01-02 06:00:00, 1591.00, 1592.50, 1590.75, 1592.25, 2317
     ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy tensorflow scikit-learn joblib tabulate
   ```

3. **Run the Code**
   ```bash
   python main_code.py
   ```

4. **Output**
   - The script saves:
     - Final dataset with actions: `es_data_with_actions.csv`
     - Training step records: `training_step_records.csv`
     - Trained model: `trading_model.h5`

---

## Troubleshooting

1. **Pandas Warning: “Setting an item of incompatible dtype”**
   - All numeric columns are cast to `float64` after computing indicators:
     ```python
     numeric_only_cols = es_data.select_dtypes(include=['number']).columns
     es_data[numeric_only_cols] = es_data[numeric_only_cols].astype('float64')
     ```

2. **NaN Loss in DQN Training**
   - Ensure rewards are clamped:
     ```python
     if reward > REWARD_CLAMP:
         reward = REWARD_CLAMP
     elif reward < -REWARD_CLAMP:
         reward = -REWARD_CLAMP
     ```
   - Use gradient clipping:
     ```python
     optimizer=Adam(learning_rate=lr, clipnorm=1.0)
     ```

3. **Joblib Worker Crashes**
   - Reduce parallelism in `calculate_bayes_prob`:
     ```python
     calculate_bayes_prob(es_data, lookback=100, n_jobs=2)
     ```

---

## Example Workflow

1. **Load Data**:
   ```python
   raw_data = load_es_data('/path/to/data.txt')
   ```

2. **Compute Indicators**:
   ```python
   es_data = compute_indicators_parallel(raw_data, chunk_size=20000, overlap=30, n_jobs=-1)
   ```

3. **Compute Bayesian Probabilities**:
   ```python
   es_data['Bayes_Prob'] = calculate_bayes_prob(es_data, lookback=100, n_jobs=2)
   ```

4. **Scale Numeric Features**:
   ```python
   es_data = partial_fit_scaler(es_data, StandardScaler(), chunk_size=10000)
   ```

5. **Train the RL Agent**:
   ```python
   agent = TradingAgent(state_dim=state_dim, action_dim=action_dim, lr=0.0005, gamma=0.95)
   agent.train()
   ```

---

## Future Improvements

1. **Enhanced Bayesian Probabilities**
   - Use a **hierarchical Bayesian model** or a **Gaussian process** for probabilities.

2. **Additional Indicators**
   - Include features like VWAP, ATR, or fundamental data.

3. **Multi-Agent Learning**
   - Train agents to trade multiple assets simultaneously.

4. **Dynamic Reward Functions**
   - Adapt reward scaling based on market volatility.

---

## License

MIT License
