"""
Evaluate a trained PPO agent on Tetris.

Loads a saved model and runs N episodes, reporting per-episode and
aggregate statistics. Optionally renders each step to the terminal.

Usage:
  uv run python rl/play.py --model models/ppo_tetris.pt
  uv run python rl/play.py --model models/ppo_tetris.pt --episodes 100 --render
  uv run python rl/play.py --random  # baseline: random agent
"""

import sys, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent / "engine" / "build" / "lib"))
import tinyrl_tetris

QUEUE_SIZE = 3
STATE_DIM  = 20*10 + 20*10 + QUEUE_SIZE*4*4 + 4*4  # 464

ACTION_NAMES = ["LEFT", "RIGHT", "DOWN", "CW", "CCW", "DROP", "SWAP", "NOOP"]


# ── Model (must match train_vec_ppo.py) ──────────────────────────────────────
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

    def act_greedy(self, x):
        """Return the greedy (argmax) action."""
        with torch.no_grad():
            logits, value = self(x)
        return logits.argmax(dim=-1).item(), value.item()

    def act_stochastic(self, x):
        """Return a sampled action + log-prob."""
        with torch.no_grad():
            logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


# ── Observation preprocessing ────────────────────────────────────────────────
def preprocess_single(obs_dict):
    """Single-env dict obs → (1, STATE_DIM) float32 tensor."""
    board  = obs_dict["board"][:20, :10].reshape(1, -1).astype(np.float32) / 7.0
    active = obs_dict["active_tetromino"][:20, :10].reshape(1, -1).astype(np.float32)
    queue  = obs_dict["queue"].reshape(1, -1).astype(np.float32) / 7.0
    holder = obs_dict["holder"].reshape(1, -1).astype(np.float32) / 7.0
    state  = np.concatenate([board, active, queue, holder], axis=1)
    return torch.from_numpy(state)


# ── Terminal board renderer ───────────────────────────────────────────────────
PIECE_CHARS = " ISZJLOT"  # index 0 = empty, 1-7 = pieces


def render_board(obs_dict, score, lines, action_name, step):
    """Print a compact ASCII view of the board (20 rows × 10 cols)."""
    board  = obs_dict["board"][:20, :10]
    active = obs_dict["active_tetromino"][:20, :10]

    rows = []
    for r in range(19, -1, -1):   # top row first
        cells = []
        for c in range(10):
            if active[r, c]:
                cells.append("[]")
            elif board[r, c]:
                cells.append("##")
            else:
                cells.append("  ")
        rows.append("|" + "".join(cells) + "|")

    print("\033[H\033[J", end="")   # clear terminal
    print(f"Step {step:5d}  Score (lines): {score:.0f}  Action: {action_name}")
    print("+" + "--"*10 + "+")
    for row in rows:
        print(row)
    print("+" + "--"*10 + "+")


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate(model_or_none, num_episodes, greedy, render, render_delay):
    env = tinyrl_tetris.TetrisEnvWT(tinyrl_tetris.STEPPED, QUEUE_SIZE)

    ep_scores  = []
    ep_lengths = []

    for ep in range(num_episodes):
        obs_dict = env.reset()
        score = 0.0
        step  = 0

        while True:
            if model_or_none is None:
                action = np.random.randint(0, 7)
            else:
                state = preprocess_single(obs_dict)
                if greedy:
                    action, _ = model_or_none.act_greedy(state)
                else:
                    action, _, _ = model_or_none.act_stochastic(state)

            if render:
                render_board(obs_dict, score, score, ACTION_NAMES[action], step)
                time.sleep(render_delay)

            obs_dict, reward, done, _ = env.step(action)
            score += reward
            step  += 1

            if done:
                break

        ep_scores.append(score)
        ep_lengths.append(step)

        if not render:
            print(f"  ep {ep+1:4d}/{num_episodes}  score={score:6.1f}  len={step:5d}")

    print()
    print("=" * 50)
    print(f"Episodes : {num_episodes}")
    print(f"Score    : mean={np.mean(ep_scores):.2f}  "
          f"std={np.std(ep_scores):.2f}  "
          f"min={np.min(ep_scores):.1f}  max={np.max(ep_scores):.1f}")
    print(f"Length   : mean={np.mean(ep_lengths):.1f}  "
          f"std={np.std(ep_lengths):.1f}  "
          f"min={np.min(ep_lengths)}  max={np.max(ep_lengths)}")
    print("=" * 50)
    return ep_scores, ep_lengths


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO Tetris agent")
    p.add_argument("--model", type=str, default=None,
                   help="Path to saved model (.pt). Omit to evaluate a random agent.")
    p.add_argument("--random", action="store_true",
                   help="Use a random agent (ignores --model)")
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of evaluation episodes (default: 20)")
    p.add_argument("--greedy", action="store_true", default=True,
                   help="Use greedy (argmax) policy (default: True)")
    p.add_argument("--stochastic", dest="greedy", action="store_false",
                   help="Use stochastic (sampled) policy")
    p.add_argument("--render", action="store_true",
                   help="Render each step to the terminal")
    p.add_argument("--render-delay", type=float, default=0.05,
                   help="Seconds between rendered frames (default: 0.05)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.random or args.model is None:
        print("Running RANDOM agent baseline...")
        model = None
    else:
        path = Path(args.model)
        if not path.exists():
            print(f"Model file not found: {path}")
            print("Train first: uv run python rl/train_vec_ppo.py")
            sys.exit(1)
        payload  = torch.load(path, map_location="cpu", weights_only=False)
        meta     = payload.get("metadata", {})
        hidden   = meta.get("hidden", 256)
        print(f"Loaded model from {path}")
        print(f"  trained for {meta.get('total_steps', '?'):,} steps  "
              f"hidden={hidden}")
        model = ActorCritic(STATE_DIM, 7, hidden)
        model.load_state_dict(payload["model_state"])
        model.eval()
        policy = "greedy" if args.greedy else "stochastic"
        print(f"  policy={policy}  episodes={args.episodes}")

    print()
    evaluate(model, args.episodes, args.greedy, args.render, args.render_delay)


if __name__ == "__main__":
    main()
