"""
Phase 1 — SAC + HER training on single-object PickPlace.

Trains a Franka Panda arm to pick the Can and place it in its container.
Uses Stable-Baselines3's built-in SAC + HER implementation.

Usage:
    source .venv/bin/activate
    python src/agents/sac/train_single.py                    # full 500k steps
    python src/agents/sac/train_single.py --steps 50000      # quick smoke test
    python src/agents/sac/train_single.py --no-wandb         # skip W&B login

Output:
    models/sac_single_best.zip   — checkpoint with best eval success rate
    models/sac_single_final.zip  — checkpoint at end of training
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--steps",    type=int, default=None,   help="override total_timesteps from config")
parser.add_argument("--no-wandb", action="store_true",      help="disable W&B logging")
parser.add_argument("--config",   default="configs/sac_single.yaml")
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

TOTAL_STEPS    = args.steps or cfg["training"]["total_timesteps"]
LEARNING_STARTS = cfg["training"]["learning_starts"]
BATCH_SIZE     = cfg["training"]["batch_size"]
LR             = cfg["sac"]["learning_rate"]
BUFFER_SIZE    = cfg["sac"]["buffer_size"]
TAU            = cfg["sac"]["tau"]
GAMMA          = cfg["sac"]["gamma"]
ENT_COEF       = cfg["sac"]["ent_coef"]
NET_ARCH       = cfg["sac"]["policy_kwargs"]["net_arch"]
N_SAMPLED_GOAL = cfg["her"]["n_sampled_goal"]
HER_STRATEGY   = cfg["her"]["goal_selection_strategy"]
EVAL_EPISODES  = cfg["eval"]["n_eval_episodes"]
EVAL_FREQ      = cfg["eval"]["eval_freq"]
BEST_MODEL_DIR  = cfg["eval"]["best_model_save_path"].strip()  # SB3 saves best_model.zip here
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
WANDB_PROJECT  = cfg["logging"]["wandb_project"]
WANDB_RUN_NAME = cfg["logging"]["wandb_run_name"]
SINGLE_OBJECT  = cfg["env"]["single_object"]
HORIZON        = cfg["env"]["horizon"]

print(f"\n{'═'*60}")
print(f"  Phase 1 — SAC + HER  ({SINGLE_OBJECT} → container)")
print(f"{'═'*60}")
print(f"  Total steps     : {TOTAL_STEPS:,}")
print(f"  Learning starts : {LEARNING_STARTS:,}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  LR              : {LR}")
print(f"  HER k           : {N_SAMPLED_GOAL}  (strategy: {HER_STRATEGY})")
print(f"  Eval every      : {EVAL_FREQ:,} steps  ×  {EVAL_EPISODES} episodes")
print(f"  Best model      : {BEST_MODEL_PATH}")

# ── W&B setup ─────────────────────────────────────────────────────────────────
use_wandb = not args.no_wandb
if use_wandb:
    try:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "total_timesteps": TOTAL_STEPS,
                "learning_rate": LR,
                "buffer_size": BUFFER_SIZE,
                "batch_size": BATCH_SIZE,
                "her_k": N_SAMPLED_GOAL,
                "her_strategy": HER_STRATEGY,
                "gamma": GAMMA,
                "tau": TAU,
                "net_arch": NET_ARCH,
                "single_object": SINGLE_OBJECT,
                "horizon": HORIZON,
            },
        )
        print(f"  W&B             : {wandb.run.url}")
    except Exception as e:
        print(f"  W&B             : DISABLED ({e})")
        use_wandb = False
else:
    print(f"  W&B             : disabled (--no-wandb)")

print()

# ── Build environment ──────────────────────────────────────────────────────────
from envs.pickplace_wrapper import PickPlaceWrapper
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

os.makedirs("models", exist_ok=True)

print("Building training environment...")
train_env = Monitor(PickPlaceWrapper(single_object=SINGLE_OBJECT, horizon=HORIZON))

print("Building eval environment...")
eval_env  = Monitor(PickPlaceWrapper(single_object=SINGLE_OBJECT, horizon=HORIZON))

# ── Custom callback: log success rate and losses to W&B ───────────────────────
class WandbCallback(BaseCallback):
    """Logs episode stats and network losses to W&B every N episodes."""

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._episode_count = 0
        self._episode_rewards = []
        self._episode_successes = []

    def _on_step(self) -> bool:
        # SB3 Monitor wraps the env and adds episode info to infos when done
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._episode_rewards.append(ep["r"])
                # success = any reward > 0 in the episode (sparse: +1 means placed)
                self._episode_successes.append(1.0 if ep["r"] > 0 else 0.0)
                self._episode_count += 1

                if self._episode_count % self.log_interval == 0 and use_wandb:
                    import wandb
                    wandb.log({
                        "train/episode_reward":  np.mean(self._episode_rewards[-self.log_interval:]),
                        "train/success_rate":    np.mean(self._episode_successes[-self.log_interval:]),
                        "train/total_episodes":  self._episode_count,
                        "train/total_steps":     self.num_timesteps,
                    })
        return True


class ProgressCallback(BaseCallback):
    """Prints a progress line every 10k steps."""

    def __init__(self, print_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            pct = 100 * self.num_timesteps / TOTAL_STEPS
            print(f"  step {self.num_timesteps:>7,} / {TOTAL_STEPS:,}  ({pct:.0f}%)")
        return True


# ── Eval callback: saves best checkpoint by eval success rate ──────────────────
class SuccessRateEvalCallback(EvalCallback):
    """
    Extends EvalCallback to also log eval success rate to W&B.
    Success = any episode reward > 0 (sparse: +1 means placed correctly).
    """

    def _on_step(self) -> bool:
        result = super()._on_step()
        # After each eval round, super() sets self.last_mean_reward
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if use_wandb:
                import wandb
                wandb.log({
                    "eval/mean_reward":   self.last_mean_reward,
                    "eval/total_steps":   self.num_timesteps,
                })
        return result


eval_callback = SuccessRateEvalCallback(
    eval_env=eval_env,
    n_eval_episodes=EVAL_EPISODES,
    eval_freq=EVAL_FREQ,
    best_model_save_path=BEST_MODEL_DIR,   # saves as {BEST_MODEL_DIR}/best_model.zip
    log_path="models/eval_logs",
    deterministic=True,
    verbose=1,
)

# ── Build SAC + HER model ──────────────────────────────────────────────────────
print("Building SAC + HER model...")
model = SAC(
    policy="MultiInputPolicy",         # handles Dict observation spaces (obs + goals)
    env=train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=N_SAMPLED_GOAL,
        goal_selection_strategy=HER_STRATEGY,
    ),
    learning_rate=LR,
    buffer_size=BUFFER_SIZE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    tau=TAU,
    gamma=GAMMA,
    ent_coef=ENT_COEF,
    policy_kwargs=dict(net_arch=NET_ARCH),
    verbose=0,
    device="auto",                     # GPU if available
)

print(f"  Policy device   : {model.device}")
print(f"  Actor params    : {sum(p.numel() for p in model.policy.actor.parameters()):,}")
print(f"  Critic params   : {sum(p.numel() for p in model.policy.critic.parameters()):,}")
print()
print(f"Training for {TOTAL_STEPS:,} steps — this takes ~2–4 hours on an RTX 4070...")
print(f"  (run with --steps 50000 for a 5-minute smoke test)\n")

# ── Train ──────────────────────────────────────────────────────────────────────
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[
        WandbCallback(log_interval=10),
        ProgressCallback(print_freq=10_000),
        eval_callback,
    ],
    log_interval=10,
    progress_bar=False,
)

# ── Save final checkpoint ──────────────────────────────────────────────────────
final_path = "models/sac_single_final"
model.save(final_path)
print(f"\n{'─'*50}")
print(f"Training complete")
print(f"  Final model  : {final_path}.zip")
print(f"  Best model   : {BEST_MODEL_PATH}")

if use_wandb:
    import wandb
    wandb.finish()
    print(f"  W&B run      : {wandb.run.url if wandb.run else 'already finished'}")
