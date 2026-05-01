"""
Phase 0 verification — run this to confirm environment setup is working.

Usage:
    source .venv/bin/activate
    python scripts/verify_phase0.py

Expected: all checks print PASS. Any FAIL means something is broken.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {label}{suffix}")
    return condition

print("\n── Phase 0: Environment Setup & Verification ──\n")

# ── 1. Imports ──────────────────────────────────────────────────────────────
print("[ Imports ]")
try:
    import robosuite; check("robosuite", True, robosuite.__version__)
except Exception as e:
    check("robosuite", False, str(e))

try:
    import mujoco; check("mujoco", True, mujoco.__version__)
except Exception as e:
    check("mujoco", False, str(e))

try:
    import torch; check("torch", True, torch.__version__)
except Exception as e:
    check("torch", False, str(e))

try:
    import stable_baselines3 as sb3; check("stable-baselines3", True, sb3.__version__)
except Exception as e:
    check("stable-baselines3", False, str(e))

try:
    import wandb; check("wandb", True, wandb.__version__)
except Exception as e:
    check("wandb", False, str(e))

# ── 2. Raw robosuite envs ────────────────────────────────────────────────────
print("\n[ Raw robosuite environments ]")
import robosuite as suite

try:
    env = suite.make("PickPlace", robots="Panda", has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True)
    obs = env.reset()
    _, r, done, _ = env.step(np.zeros(env.action_dim))
    check("PickPlace reset+step", True, f"action_dim={env.action_dim}, horizon={env.horizon}")
    check("PickPlace action dim", env.action_dim == 7, f"got {env.action_dim}, want 7")
    check("PickPlace proprio-state", obs["robot0_proprio-state"].shape == (50,),
          f"shape={obs['robot0_proprio-state'].shape}")
    env.close()
except Exception as e:
    check("PickPlace", False, str(e))

try:
    env = suite.make("NutAssembly", robots="Panda", has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True)
    obs = env.reset()
    env.step(np.zeros(env.action_dim))
    check("NutAssembly reset+step", True, f"action_dim={env.action_dim}, horizon={env.horizon}")
    check("NutAssembly proprio-state", obs["robot0_proprio-state"].shape == (50,),
          f"shape={obs['robot0_proprio-state'].shape}")
    env.close()
except Exception as e:
    check("NutAssembly", False, str(e))

# ── 3. Gym wrappers ──────────────────────────────────────────────────────────
print("\n[ Gym wrappers ]")
try:
    from envs.pickplace_wrapper import PickPlaceWrapper
    env = PickPlaceWrapper(single_object="Can", horizon=500)
    obs, _ = env.reset()
    check("PickPlaceWrapper obs shape",   obs["observation"].shape   == (64,), str(obs["observation"].shape))
    check("PickPlaceWrapper goal shape",  obs["achieved_goal"].shape == (3,),  str(obs["achieved_goal"].shape))
    check("PickPlaceWrapper action space", env.action_space.shape    == (7,),  str(env.action_space.shape))
    _, r, _, _, _ = env.step(env.action_space.sample())
    check("PickPlaceWrapper step",        r == 0.0, f"reward={r} (sparse, expected 0 on random action)")
    # HER compute_reward
    goal   = np.array([0.1, 0.2, 0.82], dtype=np.float32)
    near   = goal + np.array([0.01, 0.0, 0.0])
    far    = goal + np.array([0.50, 0.0, 0.0])
    check("PickPlaceWrapper HER reward (near)", env.compute_reward(near, goal, {}) == 1.0)
    check("PickPlaceWrapper HER reward (far)",  env.compute_reward(far,  goal, {}) == 0.0)
    env.close()
except Exception as e:
    check("PickPlaceWrapper", False, str(e))

try:
    from envs.nutassembly_wrapper import NutAssemblyWrapper
    env = NutAssemblyWrapper(single_nut="SquareNut", horizon=500)
    obs, _ = env.reset()
    check("NutAssemblyWrapper obs shape",  obs["observation"].shape   == (64,), str(obs["observation"].shape))
    check("NutAssemblyWrapper goal shape", obs["achieved_goal"].shape == (3,),  str(obs["achieved_goal"].shape))
    _, r, _, _, _ = env.step(env.action_space.sample())
    check("NutAssemblyWrapper step",       r == 0.0, f"reward={r}")
    env.close()
except Exception as e:
    check("NutAssemblyWrapper", False, str(e))

print("\n── Done ──\n")
