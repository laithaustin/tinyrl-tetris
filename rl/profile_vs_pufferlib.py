"""
Profile TinyRL C++ Tetris sim vs PufferLib C Tetris sim.

Measures:
  - Raw step throughput (steps/sec)
  - Per-step timing breakdown via cProfile
  - Observation allocation overhead
"""
import time
import cProfile
import pstats
import io
import sys
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ENGINE_LIB   = PROJECT_ROOT / "engine" / "build" / "lib"
sys.path.insert(0, str(ENGINE_LIB))

NUM_STEPS   = 100_000
WARMUP_STEPS = 1_000


# ══════════════════════════════════════════════════════════════════════════
# TinyRL benchmark helpers
# ══════════════════════════════════════════════════════════════════════════

def _run_tinyrl(num_steps: int) -> dict:
    import tinyrl_tetris
    env = tinyrl_tetris.TetrisEnv(tinyrl_tetris.STEPPED, queue_size=3)
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 7, size=num_steps + WARMUP_STEPS)

    env.reset()
    # warmup
    for i in range(WARMUP_STEPS):
        _, _, done, _ = env.step(int(actions[i]))
        if done:
            env.reset()

    env.reset()
    resets = 0
    t0 = time.perf_counter()
    for i in range(num_steps):
        _, _, done, _ = env.step(int(actions[WARMUP_STEPS + i]))
        if done:
            env.reset()
            resets += 1
    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "resets": resets}


def _run_tinyrl_wt(num_steps: int) -> dict:
    import tinyrl_tetris
    env = tinyrl_tetris.TetrisEnvWT(tinyrl_tetris.STEPPED, queue_size=3)
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 7, size=num_steps + WARMUP_STEPS)

    env.reset()
    # warmup
    for i in range(WARMUP_STEPS):
        _, _, done, _ = env.step(int(actions[i]))
        if done:
            env.reset()

    env.reset()
    resets = 0
    t0 = time.perf_counter()
    for i in range(num_steps):
        _, _, done, _ = env.step(int(actions[WARMUP_STEPS + i]))
        if done:
            env.reset()
            resets += 1
    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "resets": resets}


def benchmark_tinyrl(num_steps: int = NUM_STEPS) -> dict:
    print("\n" + "=" * 60)
    print("BENCHMARK: TinyRL C++ Engine (single env)")
    print("=" * 60)

    result = _run_tinyrl(num_steps)
    sps = num_steps / result["elapsed"]
    print(f"  Steps   : {num_steps:,}")
    print(f"  Episodes: {result['resets']}")
    print(f"  Time    : {result['elapsed']:.3f} s")
    print(f"  Speed   : {sps:,.1f} steps/sec")
    return {"name": "TinyRL C++ (single)", "steps_per_sec": sps, **result}


def benchmark_tinyrl_wt(num_steps: int = NUM_STEPS) -> dict:
    print("\n" + "=" * 60)
    print("BENCHMARK: TinyRL C++ Engine – Write-Through (single env)")
    print("=" * 60)

    result = _run_tinyrl_wt(num_steps)
    sps = num_steps / result["elapsed"]
    print(f"  Steps   : {num_steps:,}")
    print(f"  Episodes: {result['resets']}")
    print(f"  Time    : {result['elapsed']:.3f} s")
    print(f"  Speed   : {sps:,.1f} steps/sec")
    return {"name": "TinyRL C++ Write-Through", "steps_per_sec": sps, **result}


# ══════════════════════════════════════════════════════════════════════════
# PufferLib benchmark helpers
# ══════════════════════════════════════════════════════════════════════════

def _make_pufferlib_env(num_envs: int = 1):
    """Initialise a pufferlib vec-env and return (env_handle, arrays)."""
    from pufferlib.ocean.tetris import binding
    n_rows, n_cols, deck_size = 20, 10, 3
    obs_size = n_cols * n_rows + 6 + 7 * (deck_size + 1)
    obs       = np.zeros((num_envs, obs_size), dtype=np.float32)
    actions   = np.zeros(num_envs, dtype=np.int32)
    rewards   = np.zeros(num_envs, dtype=np.float32)
    terminals = np.zeros(num_envs, dtype=np.uint8)
    trunc     = np.zeros(num_envs, dtype=np.uint8)
    env = binding.vec_init(
        obs, actions, rewards, terminals, trunc,
        num_envs, 0,
        n_cols=n_cols, n_rows=n_rows, deck_size=deck_size,
    )
    return env, obs, actions, rewards, terminals, trunc


def _run_pufferlib(num_steps: int, num_envs: int = 1) -> dict:
    from pufferlib.ocean.tetris import binding
    env, obs, actions, rewards, terminals, _ = _make_pufferlib_env(num_envs)
    rng = np.random.default_rng(42)
    # Pre-generate actions so action sampling is not included in the timed loop
    pre_actions = rng.integers(0, 7, size=(num_steps + WARMUP_STEPS, num_envs), dtype=np.int32)

    binding.vec_reset(env, 0)
    # warmup
    for i in range(WARMUP_STEPS):
        actions[:] = pre_actions[i]
        binding.vec_step(env)

    binding.vec_reset(env, 0)
    total_episodes = 0
    t0 = time.perf_counter()
    for i in range(num_steps):
        actions[:] = pre_actions[WARMUP_STEPS + i]
        binding.vec_step(env)
        total_episodes += int(terminals.sum())
    elapsed = time.perf_counter() - t0
    binding.vec_close(env)
    return {"elapsed": elapsed, "resets": total_episodes}


def benchmark_pufferlib(num_steps: int = NUM_STEPS, num_envs: int = 1) -> dict:
    label = f"PufferLib C (num_envs={num_envs})"
    print("\n" + "=" * 60)
    print(f"BENCHMARK: {label}")
    print("=" * 60)

    result = _run_pufferlib(num_steps, num_envs)
    total_env_steps = num_steps * num_envs
    sps = total_env_steps / result["elapsed"]
    print(f"  Steps   : {total_env_steps:,}  ({num_steps} calls × {num_envs} envs)")
    print(f"  Episodes: {result['resets']}")
    print(f"  Time    : {result['elapsed']:.3f} s")
    print(f"  Speed   : {sps:,.1f} steps/sec")
    return {"name": label, "steps_per_sec": sps, **result}


# ══════════════════════════════════════════════════════════════════════════
# cProfile deep-dive
# ══════════════════════════════════════════════════════════════════════════

def profile_tinyrl_wt(num_steps: int = 10_000):
    import tinyrl_tetris
    env = tinyrl_tetris.TetrisEnvWT(tinyrl_tetris.STEPPED, queue_size=3)
    rng = np.random.default_rng(0)
    env.reset()

    def _inner():
        for _ in range(num_steps):
            _, _, done, _ = env.step(int(rng.integers(0, 7)))
            if done:
                env.reset()

    pr = cProfile.Profile()
    pr.enable()
    _inner()
    pr.disable()
    return pr


def profile_tinyrl(num_steps: int = 10_000):
    import tinyrl_tetris
    env = tinyrl_tetris.TetrisEnv(tinyrl_tetris.STEPPED, queue_size=3)
    rng = np.random.default_rng(0)
    env.reset()

    def _inner():
        for _ in range(num_steps):
            _, _, done, _ = env.step(int(rng.integers(0, 7)))
            if done:
                env.reset()

    pr = cProfile.Profile()
    pr.enable()
    _inner()
    pr.disable()
    return pr


def profile_pufferlib(num_steps: int = 10_000):
    from pufferlib.ocean.tetris import binding
    env, obs, actions, rewards, terminals, _ = _make_pufferlib_env(1)
    rng = np.random.default_rng(0)
    pre_actions = rng.integers(0, 7, size=num_steps, dtype=np.int32)
    binding.vec_reset(env, 0)

    def _inner():
        for i in range(num_steps):
            actions[0] = pre_actions[i]
            binding.vec_step(env)

    pr = cProfile.Profile()
    pr.enable()
    _inner()
    pr.disable()
    binding.vec_close(env)
    return pr


def print_profile(pr: cProfile.Profile, label: str, top_n: int = 15):
    print(f"\n{'─'*60}")
    print(f"cProfile: {label}  (top {top_n} by cumulative time)")
    print(f"{'─'*60}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    # strip noisy path prefixes for readability
    output = s.getvalue()
    print(output)


# ══════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════

def print_comparison(results: list[dict]):
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    valid = [r for r in results if r is not None]
    if not valid:
        print("No results.")
        return
    fastest = max(valid, key=lambda r: r["steps_per_sec"])
    print(f"{'Environment':<35} {'Steps/sec':>14} {'vs fastest':>12}")
    print("-" * 63)
    for r in valid:
        ratio = r["steps_per_sec"] / fastest["steps_per_sec"]
        tag = "baseline" if r is fastest else f"{ratio:.3f}x"
        print(f"{r['name']:<35} {r['steps_per_sec']:>14,.1f} {tag:>12}")
    print()
    print(f"Winner: {fastest['name']}  @ {fastest['steps_per_sec']:,.1f} steps/sec")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Tetris Sim Profiling: TinyRL vs PufferLib")
    print(f"Steps per benchmark: {NUM_STEPS:,}  |  Warmup: {WARMUP_STEPS:,}")
    print("=" * 60)

    results = []

    # ── throughput benchmarks ──────────────────────────────────────────────
    results.append(benchmark_tinyrl())
    results.append(benchmark_tinyrl_wt())
    results.append(benchmark_pufferlib(num_envs=1))
    results.append(benchmark_pufferlib(num_envs=16))

    print_comparison(results)

    # ── cProfile deep-dives ───────────────────────────────────────────────
    PROFILE_STEPS = 10_000
    print(f"\n{'='*60}")
    print(f"Deep profiling ({PROFILE_STEPS:,} steps each) ...")
    print("=" * 60)

    pr_tiny    = profile_tinyrl(PROFILE_STEPS)
    pr_tiny_wt = profile_tinyrl_wt(PROFILE_STEPS)
    pr_puffer  = profile_pufferlib(PROFILE_STEPS)

    print_profile(pr_tiny,    "TinyRL C++ Engine (original)")
    print_profile(pr_tiny_wt, "TinyRL C++ Engine (write-through)")
    print_profile(pr_puffer,  "PufferLib C Engine")
