# EDUCATION.md — Tutorial Log

Running tutorial document for the `rl-warehouse-pick-place` project.
Written for someone learning the stack from scratch — no assumed knowledge.

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
Stage 1: One object, one container   → easiest
Stage 2: Four objects, four containers
Stage 3: One nut, one peg
Stage 4: Two nuts, two pegs           → hardest
```

Each stage transfers skills (grasping, moving, releasing) to the next.

#### Domain Randomization
Objects start at different positions every episode. This forces the policy to generalize — it can't just memorize "the can is always at position (0.1, 0.2, 0.8)".

---

### Environment Observation and Action Spaces

#### PickPlace (Task A)
- **Observation:** `~39-dimensional` float vector — robot proprioception (joint positions + velocities) + object poses
- **Action:** `7-dimensional` float vector in `[-1, 1]` — normalized joint velocity commands
- **Goal:** Place all 4 objects in their matching containers
- **Reward:** Sparse — `+1` per object correctly placed, `0` otherwise

#### NutAssembly (Task B)
- **Observation:** `~45-dimensional` float vector — robot state + nut positions + peg positions + nut orientations
- **Action:** `7-dimensional` float vector in `[-1, 1]`
- **Goal:** Insert each nut onto the correct peg
- **Reward:** Sparse — `+1` per nut correctly assembled

---

### Architecture Diagram

```
Episode starts
      │
      ▼
High-Level DQN
  Input:  env state (which objects remain)   shape: [n_objects * 2]
  Output: sub-task index (which object/peg)  shape: [n_subtasks]
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
| `src/envs/` | Wrappers around robosuite environments |
| `src/agents/sac/` | SAC + HER low-level controller |
| `src/agents/dqn/` | High-level DQN task selector |
| `src/hierarchical/` | Combines DQN + SAC into one agent |
| `src/curriculum/` | Curriculum stage definitions |
| `src/eval/` | Evaluation harness (200+ episode runs) |
| `configs/` | Hyperparameter configs per task/stage |
