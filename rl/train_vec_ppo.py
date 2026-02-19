"""
Vectorised PPO for TinyRL Tetris using VecTetrisEnvWT.

64 environments run inside one C++ loop per Python call.
Reward shaping on top of raw line-clear reward:
  -0.05 * max(0, Δholes)   (penalise creating holes)
  -0.01 * max(0, Δmax_height)  (penalise growing the stack)
"""

import sys, time, collections, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent / "engine" / "build" / "lib"))
import tinyrl_tetris

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Vectorised PPO for TinyRL Tetris")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total environment steps (default: 2_000_000)")
    p.add_argument("--save", type=str, default="models/ppo_tetris.pt",
                   help="Path to save the final model")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Save checkpoint every N updates (0 = disabled)")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps-per-rollout", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Hyperparameters ──────────────────────────────────────────────────────────
QUEUE_SIZE     = 3
GAMMA          = 0.99
LAMBDA         = 0.95
CLIP_EPS       = 0.2
VALUE_CLIP_EPS = 0.2
ENTROPY_COEF   = 0.02   # higher entropy = more exploration
VALUE_COEF     = 0.5
EPOCHS         = 4
MINIBATCH      = 2048
MAX_GRAD_NORM  = 0.5
REPORT_EVERY   = 10   # updates between progress prints

STATE_DIM = 20*10 + 20*10 + QUEUE_SIZE*4*4 + 4*4   # 464


# ── Board metrics (for reward shaping) ──────────────────────────────────────
def board_metrics(board_obs):
    """
    board_obs : (N, 24, 18) uint8, row-0 = bottom, cols 0:10 are the playfield.
    Returns heights (N,10), holes (N,), bumpiness (N,) — all float32.
    """
    b = (board_obs[:, :20, :10] > 0)          # (N,20,10) bool, row-0 = bottom
    col_any = b.any(axis=1)                    # (N,10)
    b_flip  = b[:, ::-1, :]                    # flip: row-0 now = top
    first   = np.argmax(b_flip, axis=1)        # (N,10) index of topmost filled
    heights = np.where(col_any, 20 - first, 0).astype(np.float32)   # (N,10)

    # Holes: empty cells that have at least one filled cell above them
    cum_above = np.cumsum(b_flip, axis=1)[:, ::-1, :]  # (N,20,10)
    has_above = np.zeros_like(b, dtype=np.float32)
    has_above[:, :-1, :] = cum_above[:, 1:, :].astype(np.float32)
    holes = ((has_above > 0) & ~b).sum(axis=(1, 2)).astype(np.float32)  # (N,)

    # Bumpiness: sum of absolute adjacent column height differences
    bumpiness = np.abs(heights[:, 1:] - heights[:, :-1]).sum(axis=1)   # (N,)

    # "Almost-complete row" count: rows with ≥8 of 10 cells filled.
    # This gives a dense gradient toward actually clearing lines.
    cells_per_row = (board_obs[:, :20, :10] > 0).sum(axis=2)  # (N, 20)
    almost_full   = (cells_per_row >= 8).sum(axis=1).astype(np.float32)  # (N,)
    return heights, holes, bumpiness, almost_full


# ── Observation preprocessing ────────────────────────────────────────────────
def preprocess(obs_dict, n):
    """Flatten and normalise the dict obs to (N, STATE_DIM) float32."""
    board  = obs_dict["board"][:, :20, :10].reshape(n, -1).astype(np.float32) / 7.0
    active = obs_dict["active_tetromino"][:, :20, :10].reshape(n, -1).astype(np.float32)
    queue  = obs_dict["queue"].reshape(n, -1).astype(np.float32) / 7.0
    holder = obs_dict["holder"].reshape(n, -1).astype(np.float32) / 7.0
    return np.concatenate([board, active, queue, holder], axis=1)   # (N,464)


# ── Model ────────────────────────────────────────────────────────────────────
def ortho(in_f, out_f, std=np.sqrt(2)):
    l = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(l.weight, gain=std)
    nn.init.zeros_(l.bias)
    return l


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.shared = nn.Sequential(
            ortho(state_dim, hidden), nn.Tanh(),
            ortho(hidden,    hidden), nn.Tanh(),
        )
        self.actor  = ortho(hidden, action_dim, std=0.01)
        self.critic = ortho(hidden, 1,           std=1.0)

    def forward(self, x):
        f = self.shared(x)
        return self.actor(f), self.critic(f).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ── Generalised Advantage Estimation ─────────────────────────────────────────
def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """All inputs (T, N) numpy. Returns advantages, returns (T, N)."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    nv  = next_value.copy()
    for t in reversed(range(T)):
        mask  = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * mask - values[t]
        gae   = delta + gamma * lam * mask * gae
        adv[t] = gae
        nv = values[t]
    return adv, adv + values


def save_model(model, path, metadata=None):
    """Save model weights and optional metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    print(f"  [saved → {path}]")


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    NUM_ENVS          = args.num_envs
    STEPS_PER_ROLLOUT = args.steps_per_rollout
    TOTAL_TIMESTEPS   = args.timesteps
    NUM_UPDATES       = TOTAL_TIMESTEPS // (NUM_ENVS * STEPS_PER_ROLLOUT)

    print("=" * 65)
    print("TinyRL Tetris — Vectorised PPO")
    print(f"  envs={NUM_ENVS}  steps/rollout={STEPS_PER_ROLLOUT}  "
          f"batch={NUM_ENVS*STEPS_PER_ROLLOUT:,}")
    print(f"  total_timesteps={TOTAL_TIMESTEPS:,}  updates={NUM_UPDATES}")
    print(f"  state_dim={STATE_DIM}  hidden={args.hidden}  lr={args.lr}")
    print(f"  save_path={args.save}")
    print("=" * 65)

    env   = tinyrl_tetris.VecTetrisEnvWT(tinyrl_tetris.STEPPED, QUEUE_SIZE, NUM_ENVS)
    model = ActorCritic(STATE_DIM, 7, args.hidden)
    opt   = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Episode tracking
    ep_shaped = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_score  = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_len    = np.zeros(NUM_ENVS, dtype=np.int32)
    shaped_buf = collections.deque(maxlen=400)
    score_buf  = collections.deque(maxlen=400)
    len_buf    = collections.deque(maxlen=400)

    obs_dict  = env.reset()
    obs_np    = preprocess(obs_dict, NUM_ENVS)

    # Delta-shaping state: track per-env holes and max_height from last step
    _, prev_holes, prev_bump, _ = board_metrics(obs_dict["board"])
    prev_max_h = np.zeros(NUM_ENVS, dtype=np.float32)

    total_steps = 0
    t0 = time.perf_counter()

    print(f"\n{'Update':>7} {'Steps':>11} {'SPS':>8} "
          f"{'ep_ret':>8} {'ep_score':>9} {'ep_len':>7} {'loss':>8}")
    print("-" * 65)

    for update in range(NUM_UPDATES):
        # ── Rollout buffers ──────────────────────────────────────────────
        T, N = STEPS_PER_ROLLOUT, NUM_ENVS
        obs_buf  = np.empty((T, N, STATE_DIM), dtype=np.float32)
        act_buf  = np.empty((T, N),            dtype=np.int64)
        logp_buf = np.empty((T, N),            dtype=np.float32)
        val_buf  = np.empty((T, N),            dtype=np.float32)
        rew_buf  = np.empty((T, N),            dtype=np.float32)
        done_buf = np.empty((T, N),            dtype=np.float32)

        for step in range(T):
            obs_buf[step] = obs_np

            with torch.no_grad():
                acts, lps, _, vals = model.get_action_and_value(
                    torch.from_numpy(obs_np))

            act_np = acts.numpy().astype(np.int32)
            logp_buf[step] = lps.numpy()
            val_buf[step]  = vals.numpy()
            act_buf[step]  = act_np

            obs_dict, raw_rew, terminals = env.step(act_np)

            # Reward shaping: delta-based penalties + dense "almost full row" bonus
            is_done = terminals.astype(bool)
            heights, holes, bumpiness, almost_full = board_metrics(obs_dict["board"])
            max_h = heights.max(axis=1)

            delta_holes = holes - prev_holes          # positive = created new holes
            delta_max_h = max_h - prev_max_h          # positive = stack grew

            shaped = (raw_rew * 10.0                          # line clears dominate
                      + 0.5  * almost_full                    # bonus: rows ≥8/10 filled
                      - 0.3  * np.maximum(0, delta_holes)     # penalise new holes
                      - 0.1  * np.maximum(0, delta_max_h)     # penalise growing stack
                      - 0.005 * bumpiness)                    # mild flat-surface reward

            # Reset shaping state for envs that just terminated
            prev_holes = np.where(is_done, 0.0, holes)
            prev_max_h = np.where(is_done, 0.0, max_h)

            rew_buf[step]  = shaped
            done_buf[step] = terminals.astype(np.float32)

            obs_np = preprocess(obs_dict, NUM_ENVS)

            # Episode bookkeeping
            ep_shaped += shaped
            ep_score  += raw_rew
            ep_len    += 1
            for i in np.where(is_done)[0]:
                shaped_buf.append(ep_shaped[i])
                score_buf.append(ep_score[i])
                len_buf.append(ep_len[i])
                ep_shaped[i] = ep_score[i] = ep_len[i] = 0

        total_steps += T * N

        # Bootstrap
        with torch.no_grad():
            _, _, _, nv = model.get_action_and_value(torch.from_numpy(obs_np))
        next_val = nv.numpy()

        # GAE
        adv, ret = compute_gae(rew_buf, done_buf, val_buf, next_val, GAMMA, LAMBDA)

        # Flatten
        B = T * N
        obs_f  = torch.from_numpy(obs_buf.reshape(B, STATE_DIM))
        act_f  = torch.from_numpy(act_buf.reshape(B)).long()
        lp_f   = torch.from_numpy(logp_buf.reshape(B))
        val_f  = torch.from_numpy(val_buf.reshape(B))
        adv_f  = torch.from_numpy(adv.reshape(B))
        ret_f  = torch.from_numpy(ret.reshape(B))

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # PPO update
        loss_val = 0.0
        for _ in range(EPOCHS):
            perm = torch.randperm(B)
            for s in range(0, B, MINIBATCH):
                idx = perm[s:s + MINIBATCH]
                _, nlp, ent, nval = model.get_action_and_value(obs_f[idx], act_f[idx])

                ratio = torch.exp(nlp - lp_f[idx])
                adv_b = adv_f[idx]
                pg = -torch.min(ratio * adv_b,
                                torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_b).mean()

                v_clip = val_f[idx] + torch.clamp(nval - val_f[idx],
                                                   -VALUE_CLIP_EPS, VALUE_CLIP_EPS)
                vl = 0.5 * torch.max((nval - ret_f[idx]).pow(2),
                                      (v_clip - ret_f[idx]).pow(2)).mean()
                el = -ent.mean()

                loss = pg + VALUE_COEF * vl + ENTROPY_COEF * el
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                loss_val = loss.item()

        # Logging
        if (update + 1) % REPORT_EVERY == 0 or update == 0:
            elapsed = time.perf_counter() - t0
            sps     = total_steps / elapsed
            mr  = np.mean(shaped_buf) if shaped_buf else 0.0
            ms  = np.mean(score_buf)  if score_buf  else 0.0
            ml  = np.mean(len_buf)    if len_buf     else 0.0
            print(f"{update+1:7d} {total_steps:>11,} {sps:>8,.0f} "
                  f"{mr:>8.2f} {ms:>9.2f} {ml:>7.0f} {loss_val:>8.4f}")

        # Checkpoint
        if args.checkpoint_every and (update + 1) % args.checkpoint_every == 0:
            ckpt = Path(args.save).with_suffix(f".ckpt{update+1}.pt")
            save_model(model, str(ckpt), {"update": update+1, "total_steps": total_steps})

    elapsed = time.perf_counter() - t0
    print("-" * 65)
    print(f"\nDone: {total_steps:,} steps in {elapsed:.1f}s  "
          f"({total_steps/elapsed:,.0f} env-steps/sec end-to-end)")
    if score_buf:
        print(f"Final 400-ep stats:  "
              f"ep_score={np.mean(score_buf):.2f}  "
              f"ep_len={np.mean(len_buf):.0f}  "
              f"ep_ret={np.mean(shaped_buf):.2f}")

    # Save final model
    save_model(model, args.save, {
        "total_steps": total_steps,
        "state_dim": STATE_DIM,
        "action_dim": 7,
        "hidden": args.hidden,
        "queue_size": QUEUE_SIZE,
        "ep_score_mean": float(np.mean(score_buf)) if score_buf else 0.0,
        "ep_len_mean":   float(np.mean(len_buf))   if len_buf   else 0.0,
    })


if __name__ == "__main__":
    train()
