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
import logging
import numpy as np

# Silence robosuite's per-reset INFO spam (controller loading messages)
# WARNING-level messages (macros not found etc.) are still shown
logging.getLogger("robosuite").setLevel(logging.WARNING)

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
CHECKPOINT_DIR  = cfg["eval"]["checkpoint_dir"].strip()
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
CHECKPOINT_FREQ = max(TOTAL_STEPS // 5, 1_000)   # always 5 evenly-spaced snapshots
print(f"  Checkpoints     : every {CHECKPOINT_FREQ:,} steps → {CHECKPOINT_DIR}")

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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import KVWriter
from tqdm import tqdm

os.makedirs("models", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Building training environment...")
train_env = Monitor(PickPlaceWrapper(single_object=SINGLE_OBJECT, horizon=HORIZON))

print("Building eval environment...")
eval_env  = Monitor(PickPlaceWrapper(single_object=SINGLE_OBJECT, horizon=HORIZON))

# ── Persistent loss store: intercepts SB3 logger before it clears values ──────
class PersistentMetricsStore(KVWriter):
    """
    Added to SB3's logger output_formats so we always have the latest
    actor/critic/entropy values even after SB3 clears its own internal dict.
    """
    def __init__(self):
        self.store: dict = {}

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        self.store.update(key_values)

    def close(self) -> None:
        pass


# ── Rich progress callback: tqdm bar + live metrics postfix + W&B logging ─────
class RichProgressCallback(BaseCallback):
    """
    Single callback that does everything:
      - tqdm loading bar (shows step, %, speed, ETA)
      - live postfix on the bar: ep#, reward, success%, critic loss, entropy
      - logs all metrics to W&B every `wandb_interval` episodes
      - prints an eval result line whenever the eval callback fires

    Terminal looks like:
      Training:  20%|████      | 100k/500k [10:23<40:00, 155 steps/s,
                 ep=200, reward=0.00, success=0%, critic=0.123, entropy=2.14]

      [EVAL] step 100,000  success: 0.0%
      [EVAL] step 200,000  success: 12.0%  ← NEW BEST
    """

    def __init__(self, metrics_store: PersistentMetricsStore,
                 wandb_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._store        = metrics_store
        self._wandb_n      = wandb_interval
        self._bar          = None
        self._ep_count     = 0
        self._ep_rewards   = []
        self._ep_successes = []
        self._ep_lengths   = []

    def _on_training_start(self) -> None:
        # Logger is initialized inside learn(). With verbose=1 SB3 calls dump()
        # so our store receives values. We strip HumanOutputFormat to suppress
        # SB3's own table — our tqdm bar shows everything instead.
        from stable_baselines3.common.logger import HumanOutputFormat
        self.model.logger.output_formats = [
            fmt for fmt in self.model.logger.output_formats
            if not isinstance(fmt, HumanOutputFormat)
        ]
        self.model.logger.output_formats.append(self._store)

        approx_eps = TOTAL_STEPS // HORIZON
        self._bar = tqdm(
            total=TOTAL_STEPS,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            colour="green",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                "  {postfix}"
            ),
        )
        tqdm.write(f"  Episodes expected : ~{approx_eps:,}  (horizon={HORIZON})")
        tqdm.write(f"  Checkpoint every  : {CHECKPOINT_FREQ:,} steps\n")

    def _on_step(self) -> bool:
        self._bar.update(1)

        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._ep_rewards.append(ep["r"])
                self._ep_successes.append(1.0 if ep["r"] > 0 else 0.0)
                self._ep_lengths.append(ep["l"])
                self._ep_count += 1

                w = min(10, len(self._ep_rewards))
                mean_r   = np.mean(self._ep_rewards[-w:])
                success  = np.mean(self._ep_successes[-w:])
                s        = self._store.store  # latest losses from PersistentMetricsStore

                # Update the live tqdm postfix
                self._bar.set_postfix(ordered_dict={
                    "ep":      self._ep_count,
                    "reward":  f"{mean_r:.2f}",
                    "success": f"{success*100:.0f}%",
                    "critic":  f"{s.get('train/critic_loss', float('nan')):.3f}",
                    "actor":   f"{s.get('train/actor_loss',  float('nan')):.3f}",
                    "α":       f"{s.get('train/ent_coef',    float('nan')):.3f}",
                }, refresh=False)

                # W&B logging every N episodes
                if self._ep_count % self._wandb_n == 0 and use_wandb:
                    import wandb
                    wandb.log({
                        "train/episode_reward":  mean_r,
                        "train/success_rate":    success,
                        "train/episode_length":  np.mean(self._ep_lengths[-w:]),
                        "train/total_episodes":  self._ep_count,
                        "train/critic_loss":     s.get("train/critic_loss"),
                        "train/actor_loss":      s.get("train/actor_loss"),
                        "train/ent_coef":        s.get("train/ent_coef"),
                        "train/ent_coef_loss":   s.get("train/ent_coef_loss"),
                        "step": self.num_timesteps,
                    })

        return True

    def _on_training_end(self) -> None:
        self._bar.close()

    def notify_eval(self, step: int, success_rate: float, is_best: bool) -> None:
        """Called by SuccessRateEvalCallback to print below the tqdm bar."""
        star = "  ← NEW BEST ★" if is_best else ""
        tqdm.write(f"  [EVAL] step {step:>7,}  success: {success_rate*100:.1f}%{star}")


# ── Eval callback: saves best checkpoint + logs eval success rate ──────────────
class SuccessRateEvalCallback(EvalCallback):
    """
    Runs n_eval_episodes with a deterministic (greedy) policy, saves best checkpoint,
    and reports to the progress bar and W&B.
    """

    def __init__(self, progress_cb: "RichProgressCallback", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_cb  = progress_cb
        self._best_success = 0.0

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            success_rate = self.last_mean_reward  # sparse: mean_reward == success_rate
            is_best = success_rate > self._best_success
            if is_best:
                self._best_success = success_rate

            self._progress_cb.notify_eval(self.num_timesteps, success_rate, is_best)

            if use_wandb:
                import wandb
                wandb.log({
                    "eval/success_rate": success_rate,
                    "eval/best_success": self._best_success,
                    "step": self.num_timesteps,
                })
        return result


# Intercepts SB3 logger so losses persist after each dump() call
metrics_store = PersistentMetricsStore()

progress_cb = RichProgressCallback(metrics_store=metrics_store, wandb_interval=10)

# Saves sac_single_{N}_steps.zip at every milestone — 5 snapshots per run
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="sac_single",
    verbose=0,
)

eval_callback = SuccessRateEvalCallback(
    progress_cb=progress_cb,
    eval_env=eval_env,
    n_eval_episodes=EVAL_EPISODES,
    eval_freq=EVAL_FREQ,
    best_model_save_path=BEST_MODEL_DIR,   # saves as {BEST_MODEL_DIR}/best_model.zip
    log_path="models/eval_logs",
    deterministic=True,
    verbose=0,
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
    verbose=1,     # needed so SB3 calls logger.dump() — we strip HumanOutputFormat ourselves
    device="auto",  # GPU if available
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
        progress_cb,
        checkpoint_callback,
        eval_callback,
    ],
    log_interval=1,    # dump logger after every episode so metrics_store always has fresh values
    progress_bar=False,
)

# ── Save final checkpoint ──────────────────────────────────────────────────────
final_path = "models/sac_single_final"
model.save(final_path)
print(f"\n{'─'*50}")
print(f"Training complete")
print(f"  Final model  : {final_path}.zip")
print(f"  Best model   : {BEST_MODEL_PATH}")
# List milestone checkpoints saved
ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("sac_single")])
print(f"  Checkpoints  : {len(ckpts)} saved in {CHECKPOINT_DIR}")
for ckpt in ckpts:
    print(f"    {ckpt}")

if use_wandb:
    import wandb
    wandb.finish()
    print(f"  W&B run      : {wandb.run.url if wandb.run else 'already finished'}")
