# Reinforcement Learning: DQN vs PPO
### Gymnasium environments · PyTorch · GPU accelerated

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) For GPU support, install PyTorch with CUDA:
#    https://pytorch.org/get-started/locally/
#    Example (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Running Training

```bash
# Train both DQN and PPO on CartPole (default, 400 episodes)
python train.py --algo both --env cartpole --episodes 400

# Train only DQN on LunarLander
python train.py --algo dqn --env lunarlander --episodes 600

# Train only PPO on MountainCar
python train.py --algo ppo --env mountain --episodes 500

# Available environments:
#   cartpole    → CartPole-v1
#   lunarlander → LunarLander-v3
#   mountain    → MountainCar-v0
#   acrobot     → Acrobot-v1
```

---

## Viewing the Dashboard

Open `dashboard.html` in your browser (any modern browser).

- **Select** the environment from the dropdown
- **Click Reload** to manually refresh metrics
- **Click Auto-refresh** to watch metrics update live while training runs

The dashboard shows:
- Episode rewards (raw + smoothed)
- Training loss curves
- Cumulative reward
- DQN exploration (ε-decay)
- PPO entropy & decomposed losses
- Side-by-side algorithm comparison table

---

## GPU Usage

The script automatically uses your GPU if PyTorch detects CUDA.
You'll see a message on startup:

```
[Device] Using: cuda
[Device] GPU: NVIDIA GeForce RTX XXXX
```

If you see `Using: cpu`, your PyTorch installation may not have CUDA support.
Reinstall PyTorch with the CUDA variant from https://pytorch.org/get-started/locally/

---

## Algorithms

| Algorithm | Type | Key Hyperparameters |
|-----------|------|---------------------|
| **DQN** | Value-based, off-policy | lr=1e-3, γ=0.99, ε-greedy, replay buffer 50k, target net |
| **PPO** | Policy gradient, on-policy | lr=3e-4, γ=0.99, λ=0.95, clip ε=0.2, 4 PPO epochs |

---

## Project Structure

```
rl_project/
├── train.py          # Main training script (DQN + PPO)
├── dashboard.html    # Live metrics dashboard
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── metrics/          # Auto-created; JSON metric files saved here
    ├── DQN_CartPole_v1.json
    ├── PPO_CartPole_v1.json
    └── ...
```
