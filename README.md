**Quantile Q-Learning**

Official implementation for [Quantile Q-Learning: Revisiting Offline Extreme Q-Learning with Quantile Regression](https://arxiv.org/abs/2511.11973) [TMLR 2026].

Code is based on PyTorch.

# QQL Implementation

This implementation is based on [CORL (Clean Offline Reinforcement Learning)](https://github.com/tinkoff-ai/CORL)'s IQL code but modified for QQL (Quantile Q-Learning). If you already have a working CORL environment, you can run this directly.

## Quick Start

### Prerequisites
- **MuJoCo 2.1.0** installed at `~/.mujoco/mujoco210`
- **System paths configured** in your shell startup file (`~/.bashrc` or `~/.zshrc`):
  ```bash
  export MUJOCO_PATH=~/.mujoco/mujoco210
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
  ```

### Step 1: Create Python Environment
```bash
conda create -n qql_env python=3.8
conda activate qql_env
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Test Installation
```bash
python -c "import mujoco_py; print('MuJoCo path:', mujoco_py.utils.discover_mujoco())"
```

## Troubleshooting

### GLIBCXX_3.4.30 Error
If you encounter `ImportError: version 'GLIBCXX_3.4.30' not found`, this is a common library version conflict.

**Solution**: Force the environment to use system libraries instead of conda's older versions:

```bash
# Remove conda's old libstdc++ and link to system version
conda activate qql_env
cd $CONDA_PREFIX/lib
rm libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

# Test again
python -c "import mujoco_py; print('Success!')"
```

### Environment Reference
For reference, see `qql_env_new.yml` for a complete working environment configuration.

## Usage

### Basic Usage
```bash
python qql.py --env hopper-medium-expert-v2
```

### Configuration
All parameters are defined in `config.yaml`. QQL uses universal parameters across all environments, so no environment-specific tuning is needed.


### Training Details
- **Network Architecture**: MLP with 256 hidden units and 2 hidden layers (universal across environments)
- **Learning Rates**: 3e-4 for all networks (actor, Q-functions, value functions)
- **Batch Size**: 256 (from paper)
- **Evaluation**: Every 5000 steps with 10 episodes
- **Checkpointing**: Automatic model saving during training

## Parameter Mapping (Code -> Paper)

| Code Parameter     | Paper Notation | Description                                 |
|--------------------|----------------|---------------------------------------------|
| `discount`         | $\gamma$       | Discount factor                             |
| `tau`              | $\tau$         | Target network update rate                  |
| `beta`             | $\beta_{low}$  | Conservative estimation parameter           |
| `bc_ratio`         | $\zeta$        | Behavior cloning ratio / Policy constraint weight
| `mild`             | $\lambda$      | Mild extrapolation parameter / Gneralization coefficient
| `gamma`            | $\omega$       | Euler-Mascheroni constant
| `iql_tau_soft`     | $\alpha_1$     | Soft value function quantile                |
| `iql_tau`          | $\alpha_2$     | Optimal value function quantile             |
| `iql_tau_low`      | $\alpha_0$     | Conservative value function quantile        |

## Algorithm Overview

### QQL vs IQL
While this implementation is based on CORL's IQL code, it implements **QQL (Quantile Q-Learning)** which differs fundamentally from IQL:

1. **Quantile Regression**: Uses quantile loss instead of asymmetric L2 loss for value function learning
2. **Gap Correction**: Implements gap correction in Q-function updates for better value estimation
3. **Dual Value Functions**: Employs both optimal and soft value functions with different quantile parameters
4. **Imagination-based Updates**: Uses policy-generated actions for value function updates (Value Regulation)
5. **Combined Advantages**: Merges advantages from both value functions for policy learning

### Key Components
- **Value Functions**: Two separate networks ($V_{soft}$ and $\hat{V}$) with different quantile parameters
- **Q-Functions**: Twin Q-networks with gap correction mechanism
- **Policy**: Advantage-weighted behavior cloning with mild extrapolation
- **Imagination**: Value Regulation using policy-generated actions

## Known Issues & Disclaimer

### D4RL/MuJoCo Installation Problems
**⚠️ Warning**: Installing all dependencies can cause some difficulties, mainly due to **D4RL** and the old version of MuJoCo it is locked to. Most installation problems are related to **D4RL** and **MuJoCo**, not the QQL implementation itself.

### Disclaimer
This implementation is based on CORL's IQL code but modified for QQL methodology. Installation difficulties related to D4RL, MuJoCo, or system library conflicts are **not specific to our QQL implementation** and are common issues in the offline RL community. We recommend using existing CORL environments when possible.

## Experiment Tracking

The implementation uses **Weights & Biases (wandb)** for experiment tracking. Training metrics, evaluation scores, and model checkpoints are automatically logged. The training runs in offline mode by default, with data saved locally in the `./wandb` directory.

To sync results after reconnecting:
```bash
wandb sync wandb/offline-run-*
```

## Test Gumbel Environment Setup

For running the `test_gumbel.py` script (which demonstrates Gumbel distribution analysis corresponding to the "Experiments on β(s) Scale" in the paper), a separate lightweight environment is recommended.

### Step 1: Create Environment
```bash
conda create -n test_gumbel python=3.8
conda activate test_gumbel
```

### Step 2: Install Dependencies
```bash
pip install -r requirements_gumbel.txt
```

### Step 3: Run the Experiment
```bash
python test_gumbel.py
```

**Expected Output**: The script will generate a PDF file `toy Gumbel.pdf` with 4 subplots showing Gumbel distribution analysis for different input standard deviations, demonstrating how the scale parameter β(s) varies with input noise levels.


