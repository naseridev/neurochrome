# NeuroChrome: An Adaptive Rock-Paper-Scissors AI

## Abstract

NeuroChrome is an advanced artificial intelligence system designed to play Rock-Paper-Scissors against human opponents using machine learning techniques. The system employs a hybrid approach combining online linear models with long-term pattern recognition to predict and counter human gameplay patterns. This implementation serves as a demonstration of adaptive learning algorithms in game-theoretic environments.

## Architecture Overview

### Core Components

The system consists of several interconnected modules:

1. **Online Linear Model**: A real-time learning system that adapts to short-term patterns
2. **Long-term Pattern Counter**: A Markov-based approach for capturing extended behavioral sequences
3. **Adaptive Agent**: A meta-learning system that dynamically blends predictions from both models
4. **Unpredictability Scoring System**: A metric for evaluating player randomness and system performance

### Technical Implementation

#### Online Linear Model
```rust
struct OnlineLinearModel {
    k: usize,           // History window size
    weights: Vec<f64>,  // Model parameters
}
```

The online linear model implements a softmax regression approach with:
- **Feature Engineering**: Converts move history into one-hot encoded feature vectors
- **Real-time Learning**: Updates weights using gradient descent with learning rate adaptation
- **Regularization**: Applies weight decay (0.9999 factor) to prevent overfitting

#### Long-term Pattern Recognition
```rust
struct LongTermCounts {
    order: usize,                    // Markov order
    counts: HashMap<u64, [u32; 3]>,  // State transition counts
}
```

The pattern counter employs:
- **State Encoding**: Compresses move sequences into 64-bit integers for efficient storage
- **Bayesian Smoothing**: Adds pseudocounts to handle unseen states
- **Variable-order Markov Models**: Adapts to sequences of varying lengths

#### Adaptive Blending Strategy

The agent dynamically combines predictions using:
```rust
let blend = 0.6 - (recent_mean * 0.1) + performance_adjustment;
```

Where:
- `blend` controls the weight between long-term and short-term predictions
- `recent_mean` represents recent performance trends
- `performance_adjustment` adapts based on current game state

## Key Algorithms

### 1. Feature Extraction
The system converts the last `k` moves into a feature vector:
```
x[i*3 + move] = 1.0  // One-hot encoding for move at position i
```

### 2. Prediction Fusion
Predictions are combined using adaptive weighting:
```
combined[i] = blend * p_long[i] + (1.0 - blend) * p_short[i]
```

### 3. Action Selection
The system uses expected value maximization:
```
EV(action) = Σ P(opponent_move) × utility(action, opponent_move)
```

Where utility values are:
- Win: +1.0
- Tie: +0.2
- Loss: -1.0

### 4. Unpredictability Scoring
The UP (Unpredictability) score tracks system performance:
- **Prediction Accuracy Penalty**: Decreases when predictions are correct
- **Game Outcome Bonus**: Adjusts based on win/loss results
- **Streak Detection**: Applies additional penalties for consecutive losses

## Performance Metrics

### Unpredictability Score Categories
- **Excellent**: UP ≥ 300 (Highly unpredictable player)
- **Good**: 256 ≤ UP < 300 (Above average randomness)
- **Average**: 200 ≤ UP < 256 (Moderate predictability)
- **Poor**: UP < 200 (Highly predictable patterns)

### Adaptive Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.12 | Gradient descent step size |
| History Window | 6 | Moves considered for pattern recognition |
| Weight Decay | 0.9999 | Regularization factor |
| Blend Base | 0.6 | Default mixing ratio |
| Gamma Alpha | 0.08 | Dirichlet noise parameter |

## Installation and Usage

### Repository Download

#### Method 1: Git Clone (Recommended)
```bash
# Clone the repository
git clone https://github.com/naseridev/neurochrome.git

# Navigate to project directory
cd neurochrome
```

#### Method 2: Download ZIP
1. Go to the [GitHub repository](https://github.com/naseridev/neurochrome)
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the downloaded file
5. Navigate to the extracted folder

#### Method 3: GitHub CLI
```bash
# Using GitHub CLI
gh repo clone username/neurochrome
cd neurochrome
```

### Prerequisites
- **Rust 1.70+**: Install from [rustup.rs](https://rustup.rs/)
- **Git**: For cloning the repository
- **Terminal/Command Prompt**: For running commands

### Dependencies
The following crates will be automatically downloaded during build:
```toml
[dependencies]
rand = "0.8"
rand_distr = "0.4"
clearscreen = "2.0"
```

### Building
```bash
# Build in debug mode
cargo build

# Build optimized release version (recommended for gameplay)
cargo build --release
```

### Running
```bash
# Run debug version
cargo run

# Run release version (faster performance)
cargo run --release
```

### Game Interface
- **R/1**: Rock
- **P/2**: Paper
- **S/3**: Scissors
- **Q**: Quit game

## Research Applications

### Game Theory
- **Nash Equilibrium Analysis**: Studies optimal mixed strategies
- **Behavioral Economics**: Examines human decision-making patterns
- **Mechanism Design**: Explores incentive structures in competitive environments

### Machine Learning
- **Online Learning**: Demonstrates real-time adaptation algorithms
- **Multi-armed Bandits**: Explores exploration vs. exploitation trade-offs
- **Ensemble Methods**: Shows effective model combination techniques

### Cognitive Science
- **Pattern Recognition**: Studies human sequence learning capabilities
- **Randomness Generation**: Analyzes human pseudo-random behavior
- **Adaptive Behavior**: Examines response to changing environments

## Future Enhancements

### Advanced Learning Algorithms
- **Deep Reinforcement Learning**: Neural network-based policy learning
- **Bayesian Optimization**: Hyperparameter tuning automation
- **Meta-learning**: Learning to learn from multiple opponents

### Extended Analysis
- **Opponent Modeling**: Individual player profiling
- **Strategy Classification**: Automatic pattern categorization
- **Temporal Dynamics**: Long-term behavioral evolution tracking

### User Experience
- **Web Interface**: Browser-based gameplay
- **Statistical Dashboard**: Detailed performance analytics
- **Tournament Mode**: Multi-player competitions

## Citation

```bibtex
@software{neurochrome2024,
  title={NeuroChrome: An Adaptive Rock-Paper-Scissors AI},
  author={Anonymous},
  year={2024},
  url={https://github.com/naseridev/neurochrome},
  note={Rust implementation of adaptive game-playing AI}
}
```

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines on:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request procedures

---

*NeuroChrome demonstrates the application of machine learning techniques to classic game theory problems, providing insights into both human behavior and algorithmic adaptation strategies.*