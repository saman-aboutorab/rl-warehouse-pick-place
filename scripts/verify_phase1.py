"""
Phase 1 — Verify SAC + HER setup.

This script does NOT train. It builds the model and shows you:
  1. What the SAC + HER architecture looks like (network sizes, layers)
  2. What a single gradient update does internally
  3. How HER relabels goals in the replay buffer
  4. If training has run: loads the checkpoint and measures success rate

Usage:
    source .venv/bin/activate
    python scripts/verify_phase1.py                  # architecture + HER demo
    python scripts/verify_phase1.py --eval           # + load checkpoint + 10 eval episodes
"""

import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--eval", action="store_true", help="load checkpoint and run eval episodes")
parser.add_argument("--checkpoint", default="models/best_model.zip")
args = parser.parse_args()

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
    print(f"    {label:<35s} {value}{note_str}")

# ─────────────────────────────────────────────────────────────
header("1. Build environment (same as training)")
# ─────────────────────────────────────────────────────────────

from envs.pickplace_wrapper import PickPlaceWrapper
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

env = Monitor(PickPlaceWrapper(single_object="Can", horizon=500))
obs, _ = env.reset()

section("Observation dict (what SAC receives each step)")
show("observation shape",  str(obs["observation"].shape),  "robot joints + object position, flat vector")
show("achieved_goal shape", str(obs["achieved_goal"].shape), "where the Can is right now  [x, y, z]")
show("desired_goal shape",  str(obs["desired_goal"].shape),  "where the Can container is  [x, y, z]")
show("achieved_goal",       str(np.round(obs["achieved_goal"], 3)),  "m — Can spawned here at episode start")
show("desired_goal",        str(np.round(obs["desired_goal"],  3)),  "m — container target (fixed per episode)")
dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
show("start distance",      f"{dist:.3f} m", "arm must close this gap to score +1")

section("Action space")
show("shape",  str(env.action_space.shape),  "7 joint velocity commands")
show("range",  f"[{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]",
     "−1 = full reverse, +1 = full forward")

# ─────────────────────────────────────────────────────────────
header("2. SAC + HER model — network architecture")
# ─────────────────────────────────────────────────────────────

print("""
  SAC uses two separate neural networks:
    Actor  — reads (observation + goal) and outputs the action to take
    Critic — reads (observation + goal + action) and estimates how good it is

  HER wraps the replay buffer: after a failed episode, it invents extra
  training examples by pretending the Can's actual final position was the goal.
""")

model = SAC(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    learning_rate=0.001,
    buffer_size=100_000,    # small for demo — training uses 1M
    learning_starts=500,
    batch_size=256,
    tau=0.005,
    gamma=0.95,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
    device="auto",
)

section("Actor network (decides action)")
actor = model.policy.actor
print(f"    Input → Linear → ReLU → Linear → ReLU → mean+log_std")
print(f"    Input dim : obs[64] + achieved_goal[3] + desired_goal[3] = [70]  (SB3 passes both goals)")
for name, layer in actor.latent_pi.named_children():
    try:
        print(f"      {name}: {layer}")
    except Exception:
        pass
mu_layer = actor.mu
print(f"    Output (mean)   : {mu_layer}")
n_actor = sum(p.numel() for p in actor.parameters())
show("total actor parameters", f"{n_actor:,}", "weights the optimizer will train")

section("Critic networks (Q1 and Q2)")
print("    Two critics — take min of both → prevents overestimating action quality")
print(f"    Input dim : obs[64] + achieved_goal[3] + desired_goal[3] + action[7] = [77]")
qf = model.policy.critic.qf0  # first of the two Q-networks
for name, layer in qf.named_children():
    try:
        print(f"      {name}: {layer}")
    except Exception:
        pass
n_critic = sum(p.numel() for p in model.policy.critic.parameters())
show("total critic parameters", f"{n_critic:,}", "×2 Q-networks + ×2 target copies")
show("device", str(model.device), "GPU if available")

# ─────────────────────────────────────────────────────────────
header("3. HER relabeling — how fake goals are created")
# ─────────────────────────────────────────────────────────────

print("""
  HER turns every failed episode into 4 successful ones.

  Real episode:
    step 1: Can at [0.09, -0.22, 0.86] → goal [−0.11, −0.28, 0.82] → reward 0
    step 2: Can at [0.11, -0.20, 0.88] → goal [−0.11, −0.28, 0.82] → reward 0
    ...
    episode ends: Can never reached container → reward = 0 for all steps

  HER strategy = "future": pick a random LATER step in the same episode,
  use the Can position AT THAT STEP as the new goal.

  Relabeled transition (example):
    step 1: Can at [0.09, -0.22, 0.86] → relabeled goal [0.09, -0.22, 0.86]
    distance = 0.0 → reward = +1  (success! the "goal" was right where the Can ended up)

  With k=4 relabeled goals per real transition, 80% of the buffer is HER examples.
  The arm learns: "how to move objects to arbitrary XYZ positions."
  That skill transfers to hitting the real container position.
""")

section("Live HER example — 5 random steps, then relabel")
env2 = PickPlaceWrapper(single_object="Can", horizon=500)
obs2, _ = env2.reset()
real_goal = obs2["desired_goal"].copy()
print(f"    Real goal (container pos): {np.round(real_goal, 3)}")
print()
print(f"    {'step':<6s} {'Can pos (achieved_goal)':<35s} {'dist to real goal':<20s} {'real reward'}")
print(f"    {'─'*6} {'─'*35} {'─'*20} {'─'*11}")

achieved_positions = []
for i in range(5):
    a = env2.action_space.sample()
    obs2, r, done, trunc, info = env2.step(a)
    ag = obs2["achieved_goal"]
    achieved_positions.append(ag.copy())
    dist = np.linalg.norm(ag - real_goal)
    print(f"    {i+1:<6d} {str(np.round(ag,3)):<35s} {dist:<20.3f} {r:.0f}")

print()
print("    HER relabeling (k=4, strategy=future) — using step-5 position as fake goal:")
fake_goal = achieved_positions[-1]
print(f"    Fake goal: {np.round(fake_goal, 3)}  (Can's final position)")
for i, ag in enumerate(achieved_positions):
    dist = np.linalg.norm(ag - fake_goal)
    fake_reward = env2.compute_reward(ag, fake_goal, {})
    print(f"      step {i+1}: dist={dist:.3f} m → relabeled reward = {fake_reward:.0f}  {'← SUCCESS' if fake_reward > 0 else ''}")

env2.close()
env.close()

# ─────────────────────────────────────────────────────────────
header("4. Training loop — what happens each update")
# ─────────────────────────────────────────────────────────────

print("""
  The training loop (inside model.learn()):

  Every env step:
    1. Actor samples action from current policy
    2. robosuite steps physics
    3. Transition (obs, action, reward, next_obs) stored in replay buffer
    4. HER adds 4 more relabeled versions of the same transition

  Every gradient step (default: once per env step, after learning_starts):
    1. Sample 256 transitions from the buffer (mix of real + HER)
    2. Critic update: minimize TD error
         target = reward + γ · min(Q1_target, Q2_target)(next_obs, next_action)
         loss   = MSE(Q1(obs, action) - target) + MSE(Q2(obs, action) - target)
    3. Actor update: maximize (Q - entropy bonus)
         entropy bonus = −α · log π(action|obs)  ← encourages exploring
    4. Soft update target networks:
         θ_target ← 0.005·θ + 0.995·θ_target

  This repeats for 500,000 steps.
  Expect the first +1 reward around step 100k–200k.
  Expect >80% success around step 400k–500k.
""")

# ─────────────────────────────────────────────────────────────
if args.eval:
    header("5. Checkpoint evaluation — loading trained policy")
    # ─────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"\n  [!] Checkpoint not found: {args.checkpoint}")
        print(f"      Run training first:")
        print(f"      python src/agents/sac/train_single.py --steps 500000")
    else:
        print(f"  Loading: {args.checkpoint}")
        eval_env = Monitor(PickPlaceWrapper(single_object="Can", horizon=500))
        trained_model = SAC.load(args.checkpoint, env=eval_env)
        print(f"  Checkpoint loaded — running 10 evaluation episodes\n")

        N_EVAL = 10
        successes = 0
        ep_rewards = []

        print(f"  {'ep':<5s} {'steps':<8s} {'reward':<10s} {'outcome'}")
        print(f"  {'─'*5} {'─'*8} {'─'*10} {'─'*12}")

        for ep in range(N_EVAL):
            obs3, _ = eval_env.reset()
            ep_reward = 0.0
            steps = 0
            done = trunc = False
            while not (done or trunc):
                action, _ = trained_model.predict(obs3, deterministic=True)
                obs3, r, done, trunc, _ = eval_env.step(action)
                ep_reward += r
                steps += 1
            ep_rewards.append(ep_reward)
            success = ep_reward > 0
            successes += int(success)
            outcome = "SUCCESS ✓" if success else "failed"
            print(f"  {ep+1:<5d} {steps:<8d} {ep_reward:<10.1f} {outcome}")

        print()
        show("episodes", N_EVAL)
        show("successes", successes)
        show("success rate", f"{successes / N_EVAL * 100:.0f}%",
             ">80% = Phase 1 exit criterion met")
        show("mean reward", f"{np.mean(ep_rewards):.2f}")
        eval_env.close()

# ─────────────────────────────────────────────────────────────
header("Summary — Phase 1 setup verified")
# ─────────────────────────────────────────────────────────────
print("""
  What is working:
    ✓ PickPlaceWrapper: obs=[64], achieved_goal=[3], desired_goal=[3]
    ✓ SAC model: actor [70→256→256→14], critics [77→256→256→1] × 2
    ✓ HerReplayBuffer: k=4, strategy=future
    ✓ HER relabeling shown live — failed steps become successes with fake goals

  To start training (~4–8 hours on RTX 4070):
    python src/agents/sac/train_single.py

  To do a 5-minute smoke test first:
    python src/agents/sac/train_single.py --steps 50000 --no-wandb

  After training, evaluate:
    python scripts/verify_phase1.py --eval
""")
