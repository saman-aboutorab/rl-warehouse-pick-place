"""
Phase 0 — See what the robot environment looks like.

This script walks you through exactly what the agent sees and does:
  - What packages are installed
  - What the raw robosuite environment produces
  - How the Gym wrapper transforms that into training-ready data
  - What HER's reward function does with goals

Usage:
    source .venv/bin/activate
    python scripts/verify_phase0.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

SEP  = "─" * 60
SEP2 = "═" * 60

def header(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(f"{SEP2}")

def section(title):
    print(f"\n  ── {title} ──")

def show(label, value, note=""):
    note_str = f"  ← {note}" if note else ""
    print(f"    {label:<30s} {value}{note_str}")

# ─────────────────────────────────────────────────────────────
header("1. Installed packages")
# ─────────────────────────────────────────────────────────────

import robosuite; show("robosuite", robosuite.__version__)
import mujoco;    show("mujoco",    mujoco.__version__)
import torch;     show("torch",     torch.__version__, "GPU: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
import stable_baselines3 as sb3; show("stable-baselines3", sb3.__version__, "has SAC + HER built in")
import wandb;     show("wandb",     wandb.__version__, "experiment tracking")
import gymnasium; show("gymnasium", gymnasium.__version__)

# ─────────────────────────────────────────────────────────────
header("2. Raw robosuite — what the simulator produces")
# ─────────────────────────────────────────────────────────────
print("""
  robosuite gives us a dict of arrays on every reset() and step().
  Below you can see every key, its shape, and a sample value.
""")

import robosuite as suite

env = suite.make("PickPlace", robots="Panda", has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 use_object_obs=True)
obs = env.reset()

section("PickPlace observation dict (one episode reset)")
print(f"    {'key':<40s} {'shape':<12s} {'min':>8s} {'max':>8s}  sample")
print(f"    {'─'*40} {'─'*12} {'─'*8} {'─'*8}  ──────")
for k, v in obs.items():
    v = np.asarray(v, dtype=np.float64)
    sample = f"{v.flat[0]:.3f}" if v.size == 1 else f"[{v.flat[0]:.3f} ...]"
    print(f"    {k:<40s} {str(v.shape):<12s} {v.min():>8.3f} {v.max():>8.3f}  {sample}")

section("Action space")
lo, hi = env.action_spec
show("shape", lo.shape, "7 joint velocity commands")
show("range", f"[{lo[0]:.1f}, {hi[0]:.1f}]", "-1 = full reverse, +1 = full forward")
show("horizon", env.horizon, "max steps per episode")

section("What 5 random steps look like")
print(f"    {'step':<6s} {'reward':<10s} {'Can position (x, y, z)':<35s} {'dist to container'}")
print(f"    {'─'*6} {'─'*10} {'─'*35} {'─'*18}")
env.reset()
# get container position for Can (bin 3)
try:
    container_pos = env.target_bin_placements[3][:3]
except Exception:
    container_pos = np.array([-0.11, -0.28, 0.82])

for i in range(5):
    action = np.random.uniform(lo, hi)
    obs, reward, done, info = env.step(action)
    can_pos = np.asarray(obs["Can_pos"])
    dist = np.linalg.norm(can_pos - container_pos)
    print(f"    {i+1:<6d} {reward:<10.1f} {str(np.round(can_pos,3)):<35s} {dist:.3f} m")

env.close()

# ─────────────────────────────────────────────────────────────
header("3. Gym wrapper — what the agent actually receives")
# ─────────────────────────────────────────────────────────────
print("""
  The wrapper does three things:
    1. Flattens the obs dict → single float32 vector  (observation)
    2. Pulls out the target object's position          (achieved_goal)
    3. Pulls out the matching container's position     (desired_goal)

  This is the format Stable-Baselines3 HER expects.
""")

from envs.pickplace_wrapper import PickPlaceWrapper

section("PickPlace wrapper — single object (Can)")
env = PickPlaceWrapper(single_object="Can", horizon=500)
obs, _ = env.reset()

show("observation shape",  obs["observation"].shape,  "flat vector: proprio [50] + object-state [14]")
show("observation dtype",  obs["observation"].dtype)
show("achieved_goal",      np.round(obs["achieved_goal"], 3), "where the Can is RIGHT NOW (random each episode)")
show("desired_goal",       np.round(obs["desired_goal"],  3), "where the Can container is (fixed)")
show("distance to goal",   f"{np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f} m",
     "arm needs to close this gap")
show("action space shape", env.action_space.shape)
show("action space range", f"[{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")

section("What the first 10 elements of observation look like")
print("    (these are robot joint positions — the arm's 'body sense')")
for i, val in enumerate(obs["observation"][:10]):
    print(f"      obs[{i}] = {val:.4f}")
print("      ... (54 more values: joint velocities, eef pos/quat, gripper, object pos/quat)")

section("5 random steps through the wrapper")
print(f"    {'step':<6s} {'reward':<10s} {'achieved_goal (Can pos)':<35s} {'dist to goal'}")
print(f"    {'─'*6} {'─'*10} {'─'*35} {'─'*12}")
env.reset()
for i in range(5):
    obs, r, done, trunc, info = env.step(env.action_space.sample())
    dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    print(f"    {i+1:<6d} {r:<10.1f} {str(np.round(obs['achieved_goal'],3)):<35s} {dist:.3f} m")
env.close()

# ─────────────────────────────────────────────────────────────
header("4. HER reward function — how success is measured")
# ─────────────────────────────────────────────────────────────
print("""
  HER calls compute_reward(achieved_goal, desired_goal) to relabel
  old episodes. The function checks: is the object close enough?

  PickPlace threshold : 5 cm  (placing in a bin — some slack allowed)
""")

env = PickPlaceWrapper(single_object="Can")
env.reset()

goal = np.array([0.20, 0.40, 0.80], dtype=np.float32)
tests = [
    ("1 cm away  (inside bin)",   goal + [0.01, 0.00, 0.00]),
    ("4 cm away  (edge of bin)",  goal + [0.04, 0.00, 0.00]),
    ("6 cm away  (just outside)", goal + [0.06, 0.00, 0.00]),
    ("30 cm away (far from bin)", goal + [0.30, 0.00, 0.00]),
]

section("PickPlace (threshold = 5 cm)")
print(f"    {'scenario':<30s} {'distance':<12s} {'reward'}")
print(f"    {'─'*30} {'─'*12} {'─'*6}")
for label, achieved in tests:
    achieved = np.array(achieved, dtype=np.float32)
    dist = np.linalg.norm(achieved - goal)
    reward = env.compute_reward(achieved, goal, {})
    outcome = "+1  ✓ success" if reward == 1.0 else " 0  ✗ not there yet"
    print(f"    {label:<30s} {dist:.2f} m       {outcome}")
env.close()

# ─────────────────────────────────────────────────────────────
header("Summary — Phase 0 complete")
# ─────────────────────────────────────────────────────────────
print("""
  What is working:
    ✓ robosuite + MuJoCo running — physics simulator is live
    ✓ PickPlace env: 4 objects, 4 containers, action=[7], obs dict with 28 keys
    ✓ PickPlaceWrapper: obs=[64] single-object, obs=[106] 4-object, goal=[3]
    ✓ HER reward function: correctly returns +1 within 5 cm, 0 when far

  What you will see in Phase 1:
    → SAC + HER training on single-object PickPlace (Can → container)
    → The agent starts with reward=0.0 every episode (like the random steps above)
    → After ~200k steps it should start getting occasional +1 rewards
    → After ~500k steps it should place the Can correctly >80% of the time

  Next command:
    python scripts/verify_phase1.py   (after Phase 1 is built)
""")
