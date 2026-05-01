# EDUCATION.md — Tutorial Log

Running tutorial document for the `rl-warehouse-pick-place` project.
Written for someone learning the stack from scratch — no assumed knowledge.

---

## [2026-05-01] Phase 1: SAC + HER Training Setup

### What was built

Three files:
- [src/agents/sac/train_single.py](src/agents/sac/train_single.py) — the training loop
- [configs/sac_single.yaml](configs/sac_single.yaml) — all hyperparameters in one place
- [scripts/verify_phase1.py](scripts/verify_phase1.py) — shows the architecture and HER live

### How training works (step by step)

Each training step:
1. The **actor** network looks at `obs[64] + achieved_goal[3] + desired_goal[3]` = `[70]` and outputs a joint velocity command `[7]`
2. MuJoCo simulates one physics frame (~20ms of robot time)
3. The transition `(obs, action, reward, next_obs)` is stored in the replay buffer
4. HER immediately adds 4 more relabeled versions of the same transition (explained below)

Every gradient step (after `learning_starts=1000`):
1. Sample a batch of 256 transitions (mix of real + HER relabeled)
2. **Critic update**: compare predicted Q-values against Bellman targets, minimize MSE
3. **Actor update**: move the policy toward actions that maximize Q + entropy bonus
4. **Soft update**: slowly blend target networks toward live networks (`τ=0.005`)

### HER — what "relabeling" means

The arm almost never places the Can in the right container early in training. Without HER, reward stays 0 and the agent learns nothing.

HER's fix: after each episode, pick 4 random later steps. Use the Can's position at those steps as if they were the goal. Since the Can was actually there, reward becomes +1. This gives the agent thousands of "successes" on fake goals — but the skills it learns (grasping, moving objects) transfer to the real goal.

```
Real episode step 12: Can at [0.03, -0.39, 0.86] → real goal [-0.11, -0.28, 0.82] → reward 0
HER relabeling:       Can at [0.03, -0.39, 0.86] → fake goal [0.03, -0.39, 0.86] → reward 1 ✓
```

### Network architectures (verified by running)

**Actor** (decides action):
```
Input [70] = obs[64] + achieved_goal[3] + desired_goal[3]
→ Linear(70, 256) + ReLU
→ Linear(256, 256) + ReLU
→ mean [7] + log_std [7]  → sample action via reparameterization trick
Total: 87,566 parameters
```

**Two Critics** (estimate Q-value):
```
Input [77] = obs[64] + achieved_goal[3] + desired_goal[3] + action[7]
→ Linear(77, 256) + ReLU
→ Linear(256, 256) + ReLU
→ scalar Q-value
Two of these (Q1, Q2) — take min → prevents overestimating
Total: 172,034 parameters × 2
```

### Hyperparameters (configs/sac_single.yaml)

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 0.001 | Standard SAC default; robust across many tasks |
| `buffer_size` | 1,000,000 | Large enough to store diverse experience |
| `batch_size` | 256 | Stable gradient estimates without too much memory |
| `gamma` | 0.95 | Discounts future reward — shorter horizon tasks benefit |
| `tau` | 0.005 | Slow target network update — stabilizes training |
| `ent_coef` | auto | Self-tuning entropy keeps exploration alive |
| `her_k` | 4 | 4 relabeled goals per real transition = 80% buffer is HER |
| `her_strategy` | future | Use goals from later in same episode = informative relabeling |

### Data flow during training

```
Episode reset
    │  obs = {observation[64], achieved_goal[3], desired_goal[3]}
    ▼
Actor forward pass
    │  input: [obs + ag + dg] = [70]
    │  output: action [7]
    ▼
robosuite step
    │  reward: 0 (usually) or 1 (Can in container within 5 cm)
    ▼
Replay buffer store
    │  (obs, action, reward, next_obs) × 1 real + 4 HER relabeled
    ▼
Critic + Actor gradient update (batch of 256)
    │  critic loss: TD error
    │  actor loss: −Q + entropy bonus
    ▼
Soft-update target networks
    │  θ_target ← 0.005·θ + 0.995·θ_target
    ▼
Repeat 500,000 times
```

### What to expect during training

- Steps 0–1,000: random exploration only (learning not started yet)
- Steps 1,000–100,000: reward stays 0 but buffer fills up; HER relabeling generates "fake successes"
- Steps 100,000–200,000: first real +1 rewards appear
- Steps 300,000–500,000: success rate climbs past 50%, then past 80%

### Commands

```bash
# 5-minute smoke test (50k steps, no W&B)
python src/agents/sac/train_single.py --steps 50000 --no-wandb

# Full training (~4–8 hours, logs to W&B)
python src/agents/sac/train_single.py

# After training: evaluate the saved checkpoint
python scripts/verify_phase1.py --eval
```

---

## [2026-05-01] Deep Dive: Full Project Architecture — Agents, Data, Models

### What the project is

A robot arm learns to sort objects from scratch using only reward signals — no pre-programmed trajectories:
- **Task:** Pick 4 objects (can, cereal, milk, bread) out of a bin, place each in its matching container

The agent controls a **Franka Panda** (7-joint robot arm) inside MuJoCo via robosuite.

---

### The Environment Interface

robosuite exposes a Gym-style interface:
```python
obs = env.reset()                          # start new episode
obs, reward, done, info = env.step(action) # advance one physics step (~20ms)
# one episode = ~500 steps = ~10 seconds of robot time
```

---

### Observation Space (what the agent sees)

> Verified by running robosuite 1.5.2 — actual numbers differ from initial estimates.

Robosuite packages the full observation into two flat vectors:
- `robot0_proprio-state` — everything about the robot itself
- `object-state` — everything about the objects in the scene

**PickPlace (4-object mode)** — wrapper concatenates both:
```
robot0_proprio-state   [50]   joint pos/cos/sin/vel/acc, eef pos+quat, gripper pos+vel
object-state           [56]   pos[3]+quat[4]+rel_pos[3]+rel_quat[4] per object × 4
Total obs (wrapper):   [106]  dtype: float32

Single-object mode (e.g. one Can):
object-state           [14]   same fields, but only for the one active object
Total obs (wrapper):   [64]   dtype: float32
```

The wrapper detects these dimensions dynamically at init time, so no hardcoded numbers.

---

### Action Space (what the agent outputs)

Shape `[7]`, dtype `float64`, all values clamped to `[-1, 1]`:
```
joint_velocity_commands  [7]   one per joint
```
Note: gripper is **not** a separate action dimension in the default Panda controller —
it is embedded in the composite controller config (`default_panda.json`). The wrapper
casts to float64 before passing to robosuite, and float32 on the output side.

---

### Reward

Sparse — agent gets almost nothing until it succeeds:
```
PickPlace:  +1 per object correctly in container  (max +4 per episode)
```
This is why HER is essential — without it early training is nearly blind.

---

### HER Replay Buffer Format

Each stored transition:
```python
{
  "obs":            np.array [72],   # full observation at time t
  "action":         np.array [8],    # action taken
  "next_obs":       np.array [72],   # observation at time t+1
  "reward":         float,           # 0 or 1
  "done":           bool,
  "achieved_goal":  np.array [3],    # where the object actually ended up
  "desired_goal":   np.array [3],    # where it was supposed to go
}
```

After a failed episode, HER picks k=4 random timesteps and relabels:
`achieved_goal at step t` → treated as if it *were* the desired_goal → reward becomes +1.
The arm learns "how to move objects to arbitrary positions" before hitting the exact target.

---

### Low-Level Agent: SAC Networks

**Actor (policy)** — decides which action to take:
```
Input:   concat(obs [64], desired_goal [3])  →  [67]   (single-object mode)
         concat(obs [106], desired_goal [3]) →  [109]  (4-object mode)
Layer 1: Linear(67 → 256) + ReLU
Layer 2: Linear(256 → 256) + ReLU
Output:  mean [7] + log_std [7]  →  sample action via reparameterization
```

**Two Critics (Q1, Q2)** — estimate how good a (state, action) pair is:
```
Input:   concat(obs [64], goal [3], action [7])  →  [74]
Layer 1: Linear(74 → 256) + ReLU
Layer 2: Linear(256 → 256) + ReLU
Output:  scalar Q-value
```
Two critics, take the minimum → prevents Q-value overestimation (SAC stability trick).

**SAC training loop (every 50 env steps):**
1. Sample batch of 256 transitions from replay buffer
2. HER relabels 4 out of every 5 sampled goals
3. Update critics: minimize Bellman error
4. Update actor: maximize Q + entropy bonus (encourages exploration)
5. Soft-update target networks: `θ_target ← 0.005·θ + 0.995·θ_target`

---

### High-Level Agent: DQN Networks

Decides *which object to target next*. Runs once per sub-task (not every physics step).

**Input — high-level state** `[20]`:
```
per object (×4):
  placed_flag  [1]   1 if already correctly placed, else 0
  obj_pos      [3]   current XYZ position
Total: [4] × 4 objects = [16], plus 4 task/phase context flags = [20]
```

**Q-network:**
```
Input:   [20]
Layer 1: Linear(20 → 128) + ReLU
Layer 2: Linear(128 → 64) + ReLU
Output:  Q-value per sub-task  →  [4]  (one per object)
```

Action = sub-task index (0–3). The selected index determines which object's container position becomes the goal for SAC.

---

### Hierarchical Control Loop

```
Episode reset
     │
     ▼
DQN reads high-level state [20]
  → picks sub-task i  (which object to place next)
     │
     ▼
goal = container_i_pos  [3]
     │
     ▼
SAC runs ≤ T_low steps
  input each step: concat(obs [64], goal [3]) → [67]
  output each step: action [7]
  stops: object placed OR T_low steps elapsed
     │
     ├─ success → DQN reward +1, update DQN buffer
     └─ failure → DQN reward  0, update DQN buffer
     │
     ▼
Repeat until all objects placed or episode horizon reached
```

DQN sees sub-task outcomes. SAC sees every physics step but focuses on one object at a time.

---

### Curriculum Stages

| Stage | Task | Obs shape | Max reward |
|-------|------|-----------|------------|
| 1 | PickPlace, 1 object  | obs [64] + goal [3] = [67] | +1 |
| 2 | PickPlace, 4 objects | obs [106] + goal [3] = [109] | +4 |

Stage 2 fine-tunes from the Stage 1 checkpoint — the arm already knows how to grasp and place, it just needs to learn sequencing.

---

### Full Data Flow

```
MuJoCo physics
    │  obs dict (28 keys), reward float, done bool
    ▼
Env Wrapper  (Gym-compatible, adds achieved_goal / desired_goal)
    │  {obs [64 or 106], achieved_goal [3], desired_goal [3]}
    ▼
Replay Buffer  (capacity 1M transitions)
    │  stores raw + HER-relabeled transitions
    ▼
SAC training  (every 50 env steps, batch size 256)
    │  updates Actor [67→256→256→14] and Critics [74→256→256→1]
    ▼
Hierarchical Runner
    │  calls DQN [20→128→64→4] every T_low steps
    ▼
W&B Logger
       episode_reward, success_rate, actor_loss, critic_loss, entropy
```

---

## [2026-05-01] Project Setup: README, Structure, and Key Concepts

### What it does
This entry covers the initial project scaffold: the README, directory layout, dependency list, and core concept explanations for the algorithms used.

---

### The Big Picture

The goal is to teach a robot arm to sort objects — without telling it exactly how. Instead, it tries things, gets rewarded when it succeeds, and gradually improves. This is **Reinforcement Learning (RL)**.

Think of it like training a dog:
- The dog (robot) tries an action
- If it does something good (places the object correctly), it gets a treat (+1 reward)
- Over thousands of tries, it learns which actions lead to treats

---

### Key Concepts Explained

#### SAC — Soft Actor-Critic
An RL algorithm that controls the robot arm. It outputs **continuous actions** — meaning it says "move the joint 0.03 radians" rather than just "left/right/up/down".

**Why "soft"?** It adds a term that encourages the robot to *explore* — it doesn't just repeat the one thing that worked, it keeps trying slightly different approaches. This helps it find better solutions.

**Inputs/Outputs:**
- Input (observation): robot joint positions, velocities, object positions — a flat vector, e.g., shape `[obs_dim]` where `obs_dim ≈ 39` for PickPlace
- Output (action): joint torques or velocities — shape `[7]` (one per joint on the Panda arm)

#### HER — Hindsight Experience Replay
The sparse reward problem: the robot only gets reward when it places an object perfectly. Early in training, this almost never happens — so the robot gets almost no feedback and learns nothing.

**HER's trick:** After a failed episode, it asks "OK, the robot didn't place the can in Container 1 — but it *did* move the can to position (x, y, z). What if *that* had been the goal?" It relabels the goal and treats the episode as a success for that relabeled goal.

Over time, the robot builds up a map of "how to move objects to arbitrary positions," which it can then apply to reaching the real goal.

**Data format (experience replay buffer):**
```
(state, action, reward, next_state, goal, achieved_goal)
state:          shape [obs_dim]
action:         shape [7]
reward:         scalar (0 or 1)
goal:           shape [3]  (x, y, z target position)
achieved_goal:  shape [3]  (x, y, z position actually reached)
```

#### DQN — Deep Q-Network
The **high-level planner**. While SAC handles *how* to move, DQN handles *what to do next* — e.g., "pick up the cereal box and place it in Container 2".

DQN outputs a Q-value for each possible sub-task (discrete choices), and picks the one with the highest value.

**Inputs/Outputs:**
- Input: high-level state (which objects are placed, which remain) — shape `[n_objects * 2]` (placed/not-placed flags)
- Output: Q-value for each sub-task — shape `[n_subtasks]` (e.g., 4 objects × 4 containers = 16 options)

#### Hierarchical RL
Splits the problem into two levels:

```
High Level (DQN):   "Which object to pick next?"
                          ↓
Low Level (SAC):    "Execute the grasp-and-place motion"
```

This mirrors how humans think: you first decide "I'll grab the cereal box" (planning), then your arm figures out the exact trajectory (motor control). Combining both into one giant network causes them to interfere with each other.

#### Curriculum Learning
Training order matters. You wouldn't learn calculus before algebra. Same here:

```
Stage 1: One object, one container   → easiest — learn grasping and placing
Stage 2: Four objects, four containers → harder — learn sequencing
```

Each stage transfers skills (grasping, moving, releasing) to the next.

#### Domain Randomization
Objects start at different positions every episode. This forces the policy to generalize — it can't just memorize "the can is always at position (0.1, 0.2, 0.8)".

---

### Environment Observation and Action Spaces

#### PickPlace
- **Observation:** `[64]` (single-object) or `[106]` (4-object) float32 vector — robot proprioception + object poses
- **Action:** `[7]` float64 vector in `[-1, 1]` — normalized joint velocity commands
- **Goal:** Place all 4 objects in their matching containers
- **Reward:** Sparse — `+1` per object correctly placed, `0` otherwise

---

### Architecture Diagram

```
Episode starts
      │
      ▼
High-Level DQN
  Input:  env state (which objects remain)   shape: [n_objects * 2]
  Output: sub-task index (which object)      shape: [4]
      │
      ▼  (selected sub-task becomes goal for low level)
Low-Level SAC + HER
  Input:  observation + goal                 shape: [obs_dim + goal_dim]
  Output: joint action                       shape: [7]
      │
      ▼
robosuite / MuJoCo
  Simulates physics, returns next observation + reward
      │
      ▼
Replay Buffer (stores transitions for SAC + HER relabeling)
      │
      ▼
Training update (every N steps)
```

---

### Files Created This Session

| File | Purpose |
|------|---------|
| [README.md](README.md) | Human-facing project overview, results tables, setup instructions |
| [CLAUDE.md](CLAUDE.md) | Instructions for Claude Code — goal, permissions, conventions |
| [requirements.txt](requirements.txt) | Python dependencies |
| [.gitignore](.gitignore) | Excludes large files (models, videos, W&B logs) from git |
| [src/envs/pickplace_wrapper.py](src/envs/pickplace_wrapper.py) | Gym Dict wrapper around robosuite PickPlace |
| `src/agents/sac/` | SAC + HER low-level controller |
| `src/agents/dqn/` | High-level DQN task selector |
| `src/hierarchical/` | Combines DQN + SAC into one agent |
| `src/curriculum/` | Curriculum stage definitions |
| `src/eval/` | Evaluation harness (200+ episode runs) |
| `configs/` | Hyperparameter configs per stage |
