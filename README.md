# RL Warehouse Pick-and-Place

A hierarchical reinforcement learning system that trains a 7-DoF Panda robot arm to sort objects into the correct bins — entirely through trial and error, with no programmed motions.

---

## The Analogy

Imagine a warehouse at night. A Panda robot arm sits in front of a bin full of mixed objects — cans, cereal boxes, milk cartons, bread loaves. Next to the bin are four separate containers, one for each product type. The robot must reach in, grab each item, identify where it belongs, and place it correctly. Nobody programs the exact motions — the robot figures them out through trial and error.

---

## Environment

**Simulator:** [robosuite](https://robosuite.ai/) on [MuJoCo](https://mujoco.org/)
**Robot:** Franka Panda (7-DoF)

### Task — Bin Pick-and-Place
- Four objects (can, cereal box, milk carton, bread loaf) placed randomly in a bin each episode
- Four corresponding target containers positioned beside the bin
- Success: all four objects sorted into their correct containers

---

## Method

### Low-Level Control: SAC + HER
**SAC (Soft Actor-Critic)** is an off-policy deep RL algorithm that outputs continuous actions, making it well-suited for smooth, precise arm movements.

**HER (Hindsight Experience Replay)** addresses the sparse reward problem. The arm receives no reward until an object is fully placed correctly. HER relabels failed episodes retroactively — "you didn't place the can in Container 1, but you *did* move it to this location, so let's learn from that" — dramatically improving sample efficiency.

### High-Level Planning: DQN Selector
A high-level **DQN** agent observes the current environment state (which objects are placed, which remain) and selects the next sub-task: which object to pick next. This mirrors how real warehouse systems separate task planning from motion execution.

### Hierarchical Architecture
```
High-Level DQN
  └── selects: which object to target next
        └── Low-Level SAC
              └── executes: grasp → transport → place
```

### Curriculum Learning
Training proceeds in stages, with each stage building on skills from the previous:

1. Single-object pick-place (one can → one container)
2. Full 4-object sorting (all objects, randomised positions)

### Domain Randomization
Object positions and orientations are randomised each episode to build policies that generalise beyond the training distribution.

---

## Algorithms

| Algorithm | Role |
|-----------|------|
| SAC | Low-level continuous motor control |
| HER | Sparse reward shaping via experience relabeling |
| DQN | High-level discrete sub-task selection |
| Hierarchical RL | Decomposing planning from motion control |
| Curriculum Learning | Progressive skill acquisition across task stages |
| Domain Randomization | Robustness to object position variation |

---

## Implementation Steps

1. **Environment setup** — Install robosuite, run PickPlace with random actions, verify observation/action spaces
2. **Single-object baseline** — Train SAC + HER on one can → one container; confirm convergence in W&B
3. **Multi-object scaling** — Scale to full 4-object PickPlace; observe that flat SAC+HER struggles with sequencing
4. **Hierarchical policy** — High-level DQN selects target object, low-level SAC executes grasp-and-place
5. **Ablation study** — Compare: flat vs hierarchical, with/without curriculum, with/without domain randomisation
6. **Evaluation & demos** — 200+ episodes per config, results tables, recorded videos

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Object correctly placed in container | +1 |
| Episode success (all 4 objects placed) | +4 total |

Reward is sparse by design — the agent receives +1 only on a successful placement. HER compensates for this during replay.

---

## Project Structure

```
.
├── src/
│   ├── envs/           # robosuite Gym wrappers
│   ├── agents/
│   │   ├── sac/        # SAC + HER low-level controller
│   │   └── dqn/        # High-level DQN task selector
│   ├── hierarchical/   # Hierarchical policy combining SAC + DQN
│   ├── curriculum/     # Curriculum stage definitions
│   └── eval/           # Evaluation harness
├── configs/            # Hyperparameter configs per stage
├── scripts/            # Verify and demo scripts
├── videos/             # Recorded rollout demos
└── models/             # Saved checkpoints
```

---

## Setup

```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/verify_phase0.py

# Watch the robot (live viewer)
python scripts/view_robot.py

# Train single-object baseline
python src/agents/sac/train_single.py

# Run evaluation
python src/eval/run_eval.py --checkpoint models/sac_single_best.zip --episodes 50
```

---

## Results

> Results table and W&B report links to be added after training runs complete.

| Configuration | PickPlace Success Rate | Avg. Episode Time |
|---------------|:----------------------:|:-----------------:|
| Flat SAC+HER (single object) | — | — |
| Flat SAC+HER (4 objects) | — | — |
| Hierarchical (no curriculum) | — | — |
| Hierarchical + Curriculum | — | — |
| Hierarchical + Curriculum + DR | — | — |

---

## Demo Videos

> To be recorded after training. Run `python scripts/view_robot.py --policy models/sac_single_best.zip` to watch a trained agent.

---

## Key Design Decisions

**Why SAC over PPO?** SAC is off-policy and more sample-efficient for continuous control. With sparse reward and a long horizon, on-policy methods require far more environment interaction.

**Why HER?** Without HER, the agent almost never receives a positive reward early in training — the probability of randomly placing an object correctly is negligible. HER converts every failed episode into useful learning signal.

**Why a two-level hierarchy?** The 4-object task requires both *what to do next* (planning) and *how to do it* (motor control). A flat network tries to learn both simultaneously and struggles. Separating them lets DQN focus on object selection while SAC specialises in manipulation.

**Why curriculum?** Starting with one object lets the arm learn grasping and placing before it has to worry about sequencing. The multi-object policy fine-tunes from the single-object checkpoint rather than starting from scratch.

---

## References

- [robosuite: A Modular Simulation Framework for Robot Learning](https://robosuite.ai/)
- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [Hindsight Experience Replay (Andrychowicz et al., 2017)](https://arxiv.org/abs/1707.01495)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) — DQN
- [Curriculum Learning (Bengio et al., 2009)](https://dl.acm.org/doi/10.1145/1553374.1553380)
