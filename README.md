# RL Warehouse Pick-and-Place

A hierarchical reinforcement learning system that trains a 7-DoF Panda robot arm to sort objects into bins and assemble nuts onto pegs — entirely through trial and error, with no programmed motions.

---

## The Analogy

Imagine a warehouse at night. A Panda robot arm sits in front of a bin full of mixed objects — cans, cereal boxes, milk cartons, bread loaves. Next to the bin are four separate containers, one for each product type. The robot must reach in, grab each item, identify where it belongs, and place it correctly. Then, in the next shift, it switches to a precision task: fitting square and round nuts onto matching pegs on an assembly line. Nobody programs the exact motions — the robot figures them out through trial and error.

---

## Environment

**Simulator:** [robosuite](https://robosuite.ai/) on [MuJoCo](https://mujoco.org/)  
**Robot:** Franka Panda (7-DoF)

### Task A — Bin Pick-and-Place
- Four objects (can, cereal box, milk carton, bread loaf) placed randomly in a bin each episode
- Four corresponding target containers positioned beside the bin
- Success: all four objects sorted into their correct containers

### Task B — Nut Assembly
- Two colored pegs (square and round) fixed to the table
- Two matching nuts placed at random positions and orientations
- Success: each nut inserted onto the correct peg with fine alignment

---

## Method

### Low-Level Control: SAC + HER
**SAC (Soft Actor-Critic)** is an off-policy deep RL algorithm that outputs continuous actions, making it well-suited for smooth, precise arm movements.

**HER (Hindsight Experience Replay)** addresses the sparse reward problem. The arm receives no reward until an object is fully placed correctly. HER relabels failed episodes retroactively — "you didn't place the can in Container 1, but you *did* move it to this location, so let's learn from that" — dramatically improving sample efficiency.

### High-Level Planning: DQN Selector
A high-level **DQN** agent observes the current environment state (which objects are placed, which remain) and selects the next sub-task: which object to pick and which container to target. This mirrors how real warehouse systems separate task planning from motion execution.

### Hierarchical Architecture
```
High-Level DQN
  └── selects: which object to target next
        └── Low-Level SAC (per sub-task)
              └── executes: grasp → transport → place
```

### Curriculum Learning
Training proceeds in stages, with each stage building on skills from the previous:

1. Single-object pick-place (one can → one container)
2. Full 4-object sorting (all objects, randomized positions)
3. Single-nut assembly
4. Dual-nut assembly with orientation matching

### Domain Randomization
Object positions and orientations are randomized each episode to build policies that generalize beyond the training distribution.

---

## Algorithms

| Algorithm | Role |
|-----------|------|
| SAC | Low-level continuous motor control |
| HER | Sparse reward shaping via experience relabeling |
| DQN | High-level discrete sub-task selection |
| Hierarchical RL | Decomposing planning from motion control |
| Curriculum Learning | Progressive skill acquisition across task stages |
| Domain Randomization | Robustness to object position/orientation variation |

---

## Implementation Steps

1. **Environment setup** — Install robosuite, run `PickPlace` and `NutAssembly` envs with random actions to verify setup and understand observation/action spaces
2. **Single-object baseline** — Train SAC + HER on single-object PickPlace (one can → one container); confirm convergence in W&B within ~2 hours on a laptop
3. **Multi-object scaling** — Scale to full 4-object PickPlace; observe that vanilla SAC + HER struggles with sequencing (motivating the hierarchical approach)
4. **Hierarchical policy** — Build hierarchical agent in PyTorch: high-level DQN selects target object, low-level SAC executes grasp-and-place
5. **NutAssembly curriculum** — Add Task B; apply curriculum: single-nut first, then dual-nut with orientation matching
6. **Ablation study** — Compare: (a) flat SAC+HER vs hierarchical, (b) with/without curriculum, (c) with/without domain randomization; log all runs to W&B
7. **Evaluation harness** — Run 200+ episodes per configuration; measure success rate, completion time, and per-object accuracy
8. **Demos & reporting** — Record rollout videos from the robosuite renderer; write results tables

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Object correctly placed in container | +1 |
| Nut correctly assembled onto peg | +1 |
| Episode success (all items placed) | All sub-rewards sum |

Reward is sparse by design — the agent receives +1 only on a successful placement. HER compensates for this during replay.

---

## Project Structure

```
.
├── envs/               # robosuite environment wrappers
├── agents/
│   ├── sac/            # SAC + HER low-level controller
│   └── dqn/            # High-level DQN task selector
├── hierarchical/       # Hierarchical policy combining SAC + DQN
├── curriculum/         # Curriculum stage definitions and schedulers
├── eval/               # Evaluation harness (200+ episode runs)
├── configs/            # Hyperparameter configs per task/stage
├── videos/             # Recorded rollout demos
└── train.py            # Main training entry point
```

---

## Setup

```bash
# Install dependencies
pip install robosuite torch wandb numpy

# Verify environment
python -c "import robosuite as suite; env = suite.make('PickPlace', robots='Panda'); env.reset()"

# Train single-object baseline
python train.py --task pickplace_single --algo sac_her

# Train full hierarchical agent
python train.py --task pickplace_full --algo hierarchical --curriculum

# Run evaluation
python eval/run_eval.py --checkpoint checkpoints/hierarchical_best.pt --episodes 200
```

---

## Results

> Results table and W&B report links to be added after training runs complete.

| Configuration | PickPlace Success | NutAssembly Success | Avg. Episode Time |
|---------------|:-----------------:|:-------------------:|:-----------------:|
| Flat SAC+HER | — | — | — |
| Hierarchical (no curriculum) | — | — | — |
| Hierarchical + Curriculum | — | — | — |
| Hierarchical + Curriculum + DR | — | — | — |

---

## Demo Videos

> To be recorded from the robosuite renderer after training.

---

## Key Design Decisions

**Why SAC over PPO?** SAC is off-policy and more sample-efficient for continuous control. With a sparse reward and long horizon, on-policy methods require far more environment interaction.

**Why HER?** Without HER, the agent almost never receives a positive reward signal early in training — the probability of randomly placing an object correctly is negligible. HER converts every failed episode into useful learning signal.

**Why a two-level hierarchy?** The multi-object task requires both *what to do next* (planning) and *how to do it* (motor control). Flattening these into a single network produces a policy that tries to learn both simultaneously, which empirically struggles. Separating them lets DQN focus on object selection logic while SAC specializes in manipulation.

**Why curriculum?** The nut assembly task requires sub-millimeter alignment that the arm cannot stumble into randomly. Starting from an easier single-nut variant builds the fine motor skills that transfer to the harder dual-nut setting.

---

## References

- [robosuite: A Modular Simulation Framework for Robot Learning](https://robosuite.ai/)
- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [Hindsight Experience Replay (Andrychowicz et al., 2017)](https://arxiv.org/abs/1707.01495)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) — DQN
- [Curriculum Learning (Bengio et al., 2009)](https://dl.acm.org/doi/10.1145/1553374.1553380)
