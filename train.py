"""
Reinforcement Learning: DQN vs PPO on Gymnasium Environments
GPU accelerated via PyTorch CUDA. Metrics saved to ./metrics/.

v3 — targeted fixes per environment based on observed failure modes:
  CartPole    : already working well, minor tuning
  LunarLander : PPO frozen entropy fixed (shorter rollouts, higher ent_coef)
                DQN over-cautious fixed (reward shaping for efficiency)
  MountainCar : reward shaping wasn't enough — added curriculum start positions
                near the goal so the agent experiences success early
  Acrobot     : DQN catastrophic forgetting fixed (larger buffer, more frequent
                target updates, per-step epsilon decay instead of per-episode)
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

# ─────────────────────────── Device ──────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
print(f"[Device] Using: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

METRICS_DIR = "metrics"
MODELS_DIR  = "models"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ─────────────────────────── Utilities ───────────────────────────────────────

def save_model(algo, env_name, state_dict, meta):
    key  = f"{algo}_{env_name}".replace("-", "_")
    path = os.path.join(MODELS_DIR, f"{key}.pt")
    torch.save({"state_dict": state_dict, "meta": meta}, path)
    print(f"[Model]   Saved → {path}")

def save_metrics(algo, env_name, data):
    key = f"{algo}_{env_name}".replace("-", "_")
    with open(os.path.join(METRICS_DIR, f"{key}.json"), "w") as f:
        json.dump(data, f, indent=2)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ─────────────────────────── Reward shaping ──────────────────────────────────

def shape_reward(env_name, obs, reward, done):
    if env_name == "MountainCar-v0":
        pos, vel = obs[0], obs[1]
    
        if vel < 0 and pos > -0.6:
            reward += 0.4 * abs(vel)     
        elif vel > 0 and pos < -0.6:
            reward += 0.4 * abs(vel)    
        else:
            reward += 0.1 * abs(vel)     
    
        reward += 0.3 * max(0.0, pos - (-0.4))   
    
        if pos >= 0.45:
            reward += 10.0              

    elif env_name == "LunarLander-v3":
        y_pos = obs[1]
        vel_y = obs[3]
        if y_pos < 0.5:
            reward += 0.5 * (0.5 - y_pos)        # higher bonus closer to ground
            reward -= 0.3 * abs(vel_y)            # penalize fast descent near ground
        reward -= 0.05                            # light step penalty
    

    return reward

# ─────────────────────────── Curriculum reset (MountainCar) ──────────────────

def curriculum_reset(env, env_name, ep, total_episodes):
    obs, info = env.reset()
    if env_name == "MountainCar-v0":
        progress = ep / total_episodes
        if random.random() > progress:
            # Full range — left slope, valley, and right slope
            # so it learns the whole swing, not just the final push
            forced_pos = random.uniform(-0.8, 0.45)
            forced_vel = random.uniform(-0.04, 0.04)
            env.unwrapped.state = np.array([forced_pos, forced_vel])
            obs = env.unwrapped.state.copy()
    return obs

# ═════════════════════════════════════════════════════════════════════════════
#  Per-environment hyperparameter configs
# ═════════════════════════════════════════════════════════════════════════════

DQN_CONFIGS = {
    "CartPole-v1": dict(
        lr=1e-3, gamma=0.99, batch_size=64,
        eps_start=1.0, eps_end=0.01, eps_decay_per="episode", eps_decay=0.995,
        target_update=10, use_step_target=False,
        buffer_cap=50_000, episodes=500,
        use_lr_schedule=False,
    ),
    "LunarLander-v3": dict(
        lr=2e-4, gamma=0.95, batch_size=256, learn_start=10_000,
        eps_start=1.0, eps_end=0.02, eps_decay_per="step", eps_decay=0.995,
        target_update=15, use_step_target=True,
        buffer_cap=200_000, episodes=1000,
        reward_shaping=True,   
        use_lr_schedule=False,  
    ),
    "MountainCar-v0": dict(
        lr=1e-3, gamma=0.99, batch_size=64,
        eps_start=1.0, eps_end=0.05, eps_decay_per="step", eps_decay=0.9998,
        target_update=5, use_step_target=True,
        buffer_cap=100_000, episodes=1500,
        reward_shaping=True, curriculum=True,
        use_lr_schedule=False,
        learn_start=2000,
    ),
    "Acrobot-v1": dict(
        lr=5e-4, gamma=0.99, batch_size=128,
        eps_start=1.0, eps_end=0.01, eps_decay_per="step", eps_decay=0.9997,
        target_update=5, buffer_cap=200_000, episodes=700,
    ),
}

PPO_CONFIGS = {
    "CartPole-v1": dict(
        lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
        ppo_epochs=10, rollout_steps=1024, minibatch_size=128,
        ent_coef=0.01, vf_coef=0.5, episodes=600,
    ),
    "LunarLander-v3": dict(
        lr=2e-4,             
        gamma=0.95, lam=0.95,
        clip_eps=0.3,
        ppo_epochs=10, rollout_steps=2048, minibatch_size=256,
        ent_coef=0.05,      
        vf_coef=1.5,        
        episodes=1500,       
        reward_shaping=True,
    ),
    "MountainCar-v0": dict(
        lr=1e-4,               
        gamma=0.99, lam=0.9,   
        clip_eps=0.2,
        ppo_epochs=10,
        rollout_steps=800,
        minibatch_size=128,
        ent_coef=0.05,         
        vf_coef=0.5,
        episodes=1500,
        reward_shaping=True, curriculum=True,
    ),
    "Acrobot-v1": dict(
        lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
        ppo_epochs=10, rollout_steps=512, minibatch_size=64,
        ent_coef=0.05, vf_coef=0.5, episodes=700,
    ),
}

# ═════════════════════════════════════════════════════════════════════════════
#  Networks
# ═════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, *t): self.buf.append(t)
    def sample(self, n):
        s,a,r,s2,d = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s), dtype=torch.float32).to(DEVICE),
                torch.tensor(a, dtype=torch.long).to(DEVICE),
                torch.tensor(r, dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
                torch.tensor(d, dtype=torch.float32).to(DEVICE))
    def __len__(self): return len(self.buf)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        a    = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value

# ═════════════════════════════════════════════════════════════════════════════
#  DQN Training
# ═════════════════════════════════════════════════════════════════════════════

def train_dqn(env_name="CartPole-v1", episodes=None):
    cfg = DQN_CONFIGS.get(env_name, DQN_CONFIGS["CartPole-v1"]).copy()
    if episodes is not None: cfg["episodes"] = episodes
    do_shape     = cfg.pop("reward_shaping",  False)
    do_curric    = cfg.pop("curriculum",      False)
    decay_per    = cfg.pop("eps_decay_per",   "episode")
    use_step_tgt = cfg.pop("use_step_target", True)
    use_lr_sched = cfg.pop("use_lr_schedule", False)

    print(f"\n{'═'*60}")
    print(f"  DQN | {env_name}  ({cfg['episodes']} ep)")
    print(f"  lr={cfg['lr']}  buf={cfg['buffer_cap']}  ε-decay=per-{decay_per}")
    print(f"  shaping={do_shape}  curriculum={do_curric}")
    print(f"{'═'*60}")

    set_seed()
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net      = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["lr"])
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["episodes"], eta_min=1e-6)
             if use_lr_sched else None)
    buffer    = ReplayBuffer(cfg["buffer_cap"])
    epsilon   = cfg["eps_start"]

    metrics = {"algo":"DQN","env":env_name,
               "episode":[],"reward":[],"loss":[],"epsilon":[],"steps":[],"timestamp":[]}
    total_steps = 0
    best_avg    = -float("inf")
    best_state  = None
    t0 = time.time()

    for ep in range(1, cfg["episodes"] + 1):
        obs  = curriculum_reset(env, env_name, ep, cfg["episodes"]) if do_curric else env.reset()[0]
        ep_reward = 0.0
        ep_loss   = []
        done      = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    action = q_net(s_t).argmax(1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done      = terminated or truncated
            r_stored  = shape_reward(env_name, next_obs, reward, done) if do_shape else reward
            buffer.push(obs, action, r_stored, next_obs, float(done))
            obs        = next_obs
            ep_reward += reward          # always log true reward
            total_steps += 1

            # Per-step epsilon decay (for envs that need it)
            if decay_per == "step":
                epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

            if len(buffer) >= cfg.get("learn_start", cfg["batch_size"]):
                s, a, r, s2, d = buffer.sample(cfg["batch_size"])
                with torch.no_grad():
                    target_q = r + cfg["gamma"] * target_net(s2).max(1)[0] * (1 - d)
                curr_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss   = F.huber_loss(curr_q, target_q)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                ep_loss.append(loss.item())

            if use_step_tgt and total_steps % cfg["target_update"] == 0:
                target_net.load_state_dict(q_net.state_dict())

        # Per-episode epsilon decay
        if decay_per == "episode":
            epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        if not use_step_tgt and ep % cfg["target_update"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        if scheduler:
            scheduler.step()

        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        metrics["episode"].append(ep)
        metrics["reward"].append(ep_reward)
        metrics["loss"].append(avg_loss)
        metrics["epsilon"].append(round(epsilon, 5))
        metrics["steps"].append(total_steps)
        metrics["timestamp"].append(round(time.time() - t0, 2))

        if ep >= 50:
            ra = float(np.mean(metrics["reward"][-50:]))
            if ra > best_avg:
                best_avg  = ra
                best_state = {k: v.cpu().clone() for k, v in q_net.state_dict().items()}

        if ep % 25 == 0:
            avg25 = np.mean(metrics["reward"][-25:])
            print(f"  Ep {ep:4d} | R {ep_reward:8.1f} | Avg25 {avg25:8.1f} | ε {epsilon:.4f}")
        if ep % 10 == 0:
            save_metrics("DQN", env_name, metrics)

    final = best_state or q_net.state_dict()
    save_metrics("DQN", env_name, metrics)
    save_model("DQN", env_name, final,
               {"algo":"DQN","env":env_name,"obs_dim":obs_dim,"act_dim":act_dim,"best_reward":best_avg})
    env.close()
    print(f"\n  DQN done. Best 50-ep avg: {best_avg:.1f}")
    return metrics

# ═════════════════════════════════════════════════════════════════════════════
#  PPO Training
# ═════════════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, next_value, gamma, lam):
    adv, gae = [], 0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae   = delta + gamma * lam * (1 - d) * gae
        adv.insert(0, gae)
        next_value = v
    return adv


def train_ppo(env_name="CartPole-v1", episodes=None):
    cfg = PPO_CONFIGS.get(env_name, PPO_CONFIGS["CartPole-v1"]).copy()
    if episodes is not None: cfg["episodes"] = episodes
    do_shape  = cfg.pop("reward_shaping", False)
    do_curric = cfg.pop("curriculum",     False)

    print(f"\n{'═'*60}")
    print(f"  PPO | {env_name}  ({cfg['episodes']} ep)")
    print(f"  lr={cfg['lr']}  rollout={cfg['rollout_steps']}  ent={cfg['ent_coef']}")
    print(f"  shaping={do_shape}  curriculum={do_curric}")
    print(f"{'═'*60}")

    set_seed()
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model     = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["episodes"], eta_min=1e-7)

    metrics = {"algo":"PPO","env":env_name,
               "episode":[],"reward":[],"loss":[],"policy_loss":[],"value_loss":[],"entropy":[],"steps":[],"timestamp":[]}

    ep_count    = 0
    ep_reward   = 0.0
    total_steps = 0
    best_avg    = -float("inf")
    best_state  = None
    t0 = time.time()

    obs = (curriculum_reset(env, env_name, 0, cfg["episodes"])
           if do_curric else env.reset()[0])

    while ep_count < cfg["episodes"]:
        obs_buf,act_buf,logp_buf,val_buf,rew_buf,done_buf = [],[],[],[],[],[]

        for _ in range(cfg["rollout_steps"]):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                a, logp, _, val = model.get_action(obs_t)

            next_obs, reward, terminated, truncated, _ = env.step(a.item())
            done     = terminated or truncated
            r_stored = shape_reward(env_name, next_obs, reward, done) if do_shape else reward

            obs_buf.append(obs); act_buf.append(a.item())
            logp_buf.append(logp.item()); val_buf.append(val.item())
            rew_buf.append(r_stored); done_buf.append(float(done))

            obs         = next_obs
            ep_reward  += reward
            total_steps += 1

            if done:
                ep_count += 1
                metrics["episode"].append(ep_count)
                metrics["reward"].append(ep_reward)
                metrics["steps"].append(total_steps)
                metrics["timestamp"].append(round(time.time() - t0, 2))
                if ep_count % 25 == 0:
                    avg25 = np.mean(metrics["reward"][-25:])
                    print(f"  Ep {ep_count:4d} | R {ep_reward:8.1f} | Avg25 {avg25:8.1f}")
                if ep_count % 10 == 0:
                    save_metrics("PPO", env_name, metrics)
                ep_reward = 0.0
                obs = (curriculum_reset(env, env_name, ep_count, cfg["episodes"])
                       if do_curric else env.reset()[0])
                if ep_count >= cfg["episodes"]:
                    break

        with torch.no_grad():
            obs_t      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, nv      = model(obs_t)
            next_value = nv.item()

        advantages = compute_gae(rew_buf, val_buf, done_buf, next_value, cfg["gamma"], cfg["lam"])
        returns    = [a + v for a, v in zip(advantages, val_buf)]

        obs_t  = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(DEVICE)
        act_t  = torch.tensor(act_buf,            dtype=torch.long   ).to(DEVICE)
        logp_t = torch.tensor(logp_buf,           dtype=torch.float32).to(DEVICE)
        adv_t  = torch.tensor(advantages,         dtype=torch.float32).to(DEVICE)
        ret_t  = torch.tensor(returns,            dtype=torch.float32).to(DEVICE)
        adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_buf)
        all_pl, all_vl, all_en = [], [], []

        for _ in range(cfg["ppo_epochs"]):
            for mb in torch.randperm(n).split(cfg["minibatch_size"]):
                if len(mb) < 4: continue
                logits, vals = model(obs_t[mb])
                dist     = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy  = dist.entropy().mean()
                ratio    = (new_logp - logp_t[mb]).exp()
                s1 = ratio * adv_t[mb]
                s2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_t[mb]
                pl = -torch.min(s1, s2).mean()
                vl = F.mse_loss(vals.squeeze(), ret_t[mb])
                loss = pl + cfg["vf_coef"] * vl - cfg["ent_coef"] * entropy
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                all_pl.append(pl.item()); all_vl.append(vl.item()); all_en.append(entropy.item())

        scheduler.step()

        avg_pl = float(np.mean(all_pl)) if all_pl else 0.0
        avg_vl = float(np.mean(all_vl)) if all_vl else 0.0
        avg_en = float(np.mean(all_en)) if all_en else 0.0
        missing = len(metrics["episode"]) - len(metrics.get("loss", []))
        for _ in range(max(0, missing)):
            metrics.setdefault("loss", []).append(avg_pl + cfg["vf_coef"] * avg_vl)
            metrics.setdefault("policy_loss", []).append(avg_pl)
            metrics.setdefault("value_loss", []).append(avg_vl)
            metrics.setdefault("entropy", []).append(avg_en)

        if len(metrics["reward"]) >= 50:
            ra = float(np.mean(metrics["reward"][-50:]))
            if ra > best_avg:
                best_avg  = ra
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    final = best_state or model.state_dict()
    save_metrics("PPO", env_name, metrics)
    save_model("PPO", env_name, final,
               {"algo":"PPO","env":env_name,"obs_dim":obs_dim,"act_dim":act_dim,"best_reward":best_avg})
    env.close()
    print(f"\n  PPO done. Best 50-ep avg: {best_avg:.1f}")
    return metrics

# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

ENVS = {
    "cartpole":    "CartPole-v1",
    "lunarlander": "LunarLander-v3",
    "mountain":    "MountainCar-v0",
    "acrobot":     "Acrobot-v1",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",     choices=["dqn","ppo","both"], default="both")
    parser.add_argument("--env",      choices=list(ENVS.keys()),    default="cartpole")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count (each env has a tuned default)")
    args = parser.parse_args()

    env_name = ENVS[args.env]
    print(f"\n[Config] env={env_name}  device={DEVICE}")

    if args.algo in ("dqn", "both"): train_dqn(env_name, args.episodes)
    if args.algo in ("ppo", "both"): train_ppo(env_name, args.episodes)

    print("\n[Done] Open http://localhost:8765 to view results.")