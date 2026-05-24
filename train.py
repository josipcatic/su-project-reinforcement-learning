"""
Reinforcement Learning: DQN vs PPO on Gymnasium Environments
Supports GPU acceleration via PyTorch CUDA.
Metrics are saved to ./metrics/ for the dashboard to visualize.
"""

import os
import json
import time
import random
import argparse
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym

# ─────────────────────────── Device Setup ────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

METRICS_DIR = "metrics"
MODELS_DIR  = "models"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ─────────────────────────── Utilities ───────────────────────────────────────

def save_model(algo, env_name, state_dict, meta: dict):
    key  = f"{algo}_{env_name}".replace("-", "_")
    path = os.path.join(MODELS_DIR, f"{key}.pt")
    torch.save({"state_dict": state_dict, "meta": meta}, path)
    print(f"[Model]   Saved → {path}")


def save_metrics(algo, env_name, data: dict):
    key = f"{algo}_{env_name}".replace("-", "_")
    path = os.path.join(METRICS_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Metrics] Saved → {path}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
#  DEEP Q-NETWORK (DQN)
# ═════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32).to(DEVICE),
            torch.tensor(a,            dtype=torch.long   ).to(DEVICE),
            torch.tensor(r,            dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
            torch.tensor(d,            dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


def train_dqn(env_name="CartPole-v1", episodes=500, lr=1e-3,
              gamma=0.99, batch_size=64, eps_start=1.0,
              eps_end=0.01, eps_decay=0.995, target_update=10):

    print(f"\n{'═'*60}")
    print(f"  DQN | {env_name}")
    print(f"{'═'*60}")

    set_seed()
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net     = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer    = ReplayBuffer()
    epsilon   = eps_start

    metrics = {
        "algo": "DQN", "env": env_name,
        "episode": [], "reward": [], "loss": [],
        "epsilon": [], "steps": [], "timestamp": []
    }

    total_steps = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        ep_loss   = []
        done      = False

        while not done:
            # ε-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    action = q_net(s_t).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward  += reward
            total_steps += 1

            # Learn
            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)

                with torch.no_grad():
                    target_q = r + gamma * target_net(s2).max(1)[0] * (1 - d)

                current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()
                ep_loss.append(loss.item())

        # Decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)

        # Update target network
        if ep % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        metrics["episode"].append(ep)
        metrics["reward"].append(ep_reward)
        metrics["loss"].append(avg_loss)
        metrics["epsilon"].append(round(epsilon, 4))
        metrics["steps"].append(total_steps)
        metrics["timestamp"].append(round(time.time() - t0, 2))

        if ep % 25 == 0:
            recent = np.mean(metrics["reward"][-25:])
            print(f"  Ep {ep:4d} | Reward {ep_reward:7.1f} | "
                  f"Avg(25) {recent:7.1f} | ε={epsilon:.3f} | Loss={avg_loss:.4f}")

        # Save metrics every 10 episodes for live dashboard
        if ep % 10 == 0:
            save_metrics("DQN", env_name, metrics)

    save_metrics("DQN", env_name, metrics)
    save_model("DQN", env_name, q_net.state_dict(), {
        "algo": "DQN", "env": env_name,
        "obs_dim": obs_dim, "act_dim": act_dim,
        "best_reward": float(max(metrics["reward"])),
    })
    env.close()
    print(f"\n  DQN training complete. Best reward: {max(metrics['reward']):.1f}")
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
#  PROXIMAL POLICY OPTIMIZATION (PPO)
# ═════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae   = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages


def train_ppo(env_name="CartPole-v1", episodes=500, lr=3e-4,
              gamma=0.99, lam=0.95, clip_eps=0.2,
              ppo_epochs=4, rollout_steps=512, minibatch_size=64,
              ent_coef=0.01, vf_coef=0.5):

    print(f"\n{'═'*60}")
    print(f"  PPO | {env_name}")
    print(f"{'═'*60}")

    set_seed()
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model     = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    metrics = {
        "algo": "PPO", "env": env_name,
        "episode": [], "reward": [], "loss": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
        "steps": [], "timestamp": []
    }

    obs, _     = env.reset()
    ep_reward  = 0
    ep_count   = 0
    total_steps = 0
    ep_rewards  = []
    t0 = time.time()

    while ep_count < episodes:
        # ── Collect rollout ────────────────────────────────────────────────
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

        for _ in range(rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, logp, _, value = model.get_action(obs_t)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action.item())
            logp_buf.append(logp.item())
            val_buf.append(value.item())
            rew_buf.append(reward)
            done_buf.append(float(done))

            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            if done:
                ep_count  += 1
                ep_rewards.append(ep_reward)

                if ep_count % 25 == 0:
                    avg = np.mean(ep_rewards[-25:])
                    print(f"  Ep {ep_count:4d} | Reward {ep_reward:7.1f} | Avg(25) {avg:7.1f}")

                # Record metrics
                metrics["episode"].append(ep_count)
                metrics["reward"].append(ep_reward)
                metrics["steps"].append(total_steps)
                metrics["timestamp"].append(round(time.time() - t0, 2))
                ep_reward = 0

                if ep_count % 10 == 0:
                    # Placeholder losses until we update below
                    save_metrics("PPO", env_name, metrics)

                obs, _ = env.reset()
                if ep_count >= episodes:
                    break

        # Bootstrap value
        with torch.no_grad():
            obs_t      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, nv      = model(obs_t)
            next_value = nv.item()

        advantages = compute_gae(rew_buf, val_buf, done_buf, next_value, gamma, lam)
        returns    = [a + v for a, v in zip(advantages, val_buf)]

        # Convert to tensors
        obs_t  = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(DEVICE)
        act_t  = torch.tensor(act_buf,            dtype=torch.long   ).to(DEVICE)
        logp_t = torch.tensor(logp_buf,           dtype=torch.float32).to(DEVICE)
        adv_t  = torch.tensor(advantages,         dtype=torch.float32).to(DEVICE)
        ret_t  = torch.tensor(returns,            dtype=torch.float32).to(DEVICE)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── PPO update ─────────────────────────────────────────────────────
        n = len(obs_buf)
        all_pol_loss, all_val_loss, all_ent = [], [], []

        for _ in range(ppo_epochs):
            idxs = torch.randperm(n)
            for start in range(0, n, minibatch_size):
                mb = idxs[start:start + minibatch_size]

                logits, values = model(obs_t[mb])
                dist    = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy  = dist.entropy().mean()

                ratio = (new_logp - logp_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(values.squeeze(), ret_t[mb])
                loss     = pol_loss + vf_coef * val_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                all_pol_loss.append(pol_loss.item())
                all_val_loss.append(val_loss.item())
                all_ent.append(entropy.item())

        # Append losses to last batch of episodes recorded this rollout
        n_new = len(metrics["episode"]) - len(metrics.get("loss", []))
        avg_pl = float(np.mean(all_pol_loss))
        avg_vl = float(np.mean(all_val_loss))
        avg_en = float(np.mean(all_ent))

        # Pad losses to align with episode metrics
        for _ in range(max(0, len(metrics["episode"]) - len(metrics.get("loss", [])))):
            metrics.setdefault("loss", []).append(avg_pl + vf_coef * avg_vl)
            metrics.setdefault("policy_loss", []).append(avg_pl)
            metrics.setdefault("value_loss", []).append(avg_vl)
            metrics.setdefault("entropy", []).append(avg_en)

    save_metrics("PPO", env_name, metrics)
    save_model("PPO", env_name, model.state_dict(), {
        "algo": "PPO", "env": env_name,
        "obs_dim": obs_dim, "act_dim": act_dim,
        "best_reward": float(max(metrics["reward"])),
    })
    env.close()
    print(f"\n  PPO training complete. Best reward: {max(metrics['reward']):.1f}")
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

ENVS = {
    "cartpole":    "CartPole-v1",
    "lunarlander": "LunarLander-v3",
    "mountain":    "MountainCar-v0",
    "acrobot":     "Acrobot-v1",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training: DQN vs PPO")
    parser.add_argument("--algo",  choices=["dqn","ppo","both"], default="both",
                        help="Algorithm to run")
    parser.add_argument("--env",   choices=list(ENVS.keys()),    default="cartpole",
                        help="Environment shortname")
    parser.add_argument("--episodes", type=int, default=400,
                        help="Number of training episodes")
    args = parser.parse_args()

    env_name = ENVS[args.env]
    print(f"\n[Config] env={env_name}, episodes={args.episodes}, device={DEVICE}")

    if args.algo in ("dqn", "both"):
        train_dqn(env_name=env_name, episodes=args.episodes)

    if args.algo in ("ppo", "both"):
        train_ppo(env_name=env_name, episodes=args.episodes)

    print("\n[Done] All training complete. Open dashboard.html to view results.")