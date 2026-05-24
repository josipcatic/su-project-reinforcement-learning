"""
play.py — Watch a trained agent play a Gymnasium environment.

Usage:
    python play.py --algo dqn --env cartpole
    python play.py --algo ppo --env lunarlander
    python play.py --algo dqn --env cartpole --episodes 5 --slow

Arguments:
    --algo      dqn | ppo
    --env       cartpole | lunarlander | mountain | acrobot
    --episodes  how many episodes to watch (default: 3)
    --slow      add a small delay between frames so you can follow the action
    --no-render just run headlessly and print stats (useful for benchmarking)
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENVS = {
    "cartpole":    "CartPole-v1",
    "lunarlander": "LunarLander-v3",
    "mountain":    "MountainCar-v0",
    "acrobot":     "Acrobot-v1",
}


# ─────────────────────── Network definitions (must match train.py) ────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def best_action(self, obs_t):
        logits, value = self(obs_t)
        return logits.argmax(dim=-1).item(), value.item()


# ─────────────────────────── Load model ──────────────────────────────────────

def load_model(algo, env_name):
    key  = f"{algo}_{env_name}".replace("-", "_")
    path = os.path.join(MODELS_DIR, f"{key}.pt")

    if not os.path.isfile(path):
        print(f"\n  [Error] No saved model found at: {path}")
        print(f"  Train first with:  python train.py --algo {algo.lower()} --env <env>\n")
        raise SystemExit(1)

    ckpt     = torch.load(path, map_location=DEVICE, weights_only=False)
    meta     = ckpt["meta"]
    obs_dim  = meta["obs_dim"]
    act_dim  = meta["act_dim"]
    best_rew = meta.get("best_reward", "?")

    print(f"\n  [Model] Loaded  : {path}")
    print(f"  [Model] Algo    : {algo.upper()}")
    print(f"  [Model] Env     : {env_name}")
    print(f"  [Model] Best rew: {best_rew:.1f}" if isinstance(best_rew, float) else f"  [Model] Best rew: {best_rew}")

    if algo.upper() == "DQN":
        net = QNetwork(obs_dim, act_dim).to(DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net, "DQN"
    else:
        net = ActorCritic(obs_dim, act_dim).to(DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net, "PPO"


# ─────────────────────────── Action selection ─────────────────────────────────

def select_action(net, algo, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if algo == "DQN":
            return net(obs_t).argmax(dim=1).item()
        else:
            return net.best_action(obs_t)[0]


# ─────────────────────────── Play loop ───────────────────────────────────────

def play(algo, env_name, episodes=3, slow=False, render=True):
    net, algo = load_model(algo, env_name)

    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    print(f"\n{'═'*55}")
    print(f"  Watching {algo} play {env_name}  ({episodes} episode{'s' if episodes>1 else ''})")
    print(f"{'═'*55}\n")

    all_rewards = []
    all_lengths = []

    for ep in range(1, episodes + 1):
        obs, _   = env.reset()
        ep_reward = 0
        steps     = 0
        done      = False

        while not done:
            action = select_action(net, algo, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            ep_reward += reward
            steps     += 1

            if slow:
                time.sleep(0.03)   # ~30 fps feel

        all_rewards.append(ep_reward)
        all_lengths.append(steps)
        print(f"  Episode {ep:2d}  |  Reward: {ep_reward:8.2f}  |  Steps: {steps}")

    env.close()

    print(f"\n{'─'*55}")
    print(f"  Summary over {episodes} episode{'s' if episodes>1 else ''}:")
    print(f"    Mean reward : {np.mean(all_rewards):.2f}")
    print(f"    Best reward : {np.max(all_rewards):.2f}")
    print(f"    Mean steps  : {np.mean(all_lengths):.1f}")
    print(f"{'─'*55}\n")


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained RL agent play")
    parser.add_argument("--algo",     choices=["dqn","ppo"], required=True,
                        help="Which algorithm's model to load")
    parser.add_argument("--env",      choices=list(ENVS.keys()), required=True,
                        help="Environment to play")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to watch (default: 3)")
    parser.add_argument("--slow",     action="store_true",
                        help="Add frame delay so you can follow the action")
    parser.add_argument("--no-render", dest="render", action="store_false",
                        help="Run headlessly (no window), just print stats")
    args = parser.parse_args()

    env_name = ENVS[args.env]
    play(
        algo     = args.algo.upper(),
        env_name = env_name,
        episodes = args.episodes,
        slow     = args.slow,
        render   = args.render,
    )