# Project Phases & Steps

Roadmap for `rl-warehouse-pick-place`. Work through phases in order — each one builds on the last.
Update the checkboxes as steps are completed.

---

## Phase 0 — Environment Setup & Verification
**Goal:** Get the simulator running and understand what data the robot sees and produces.
**Exit criterion:** Both robosuite environments run without error; observation and action shapes are logged.

- [ ] **0.1** Install dependencies: `pip install -r requirements.txt`
- [ ] **0.2** Smoke-test robosuite: instantiate `PickPlace` and `NutAssembly` envs, call `reset()` and `step()` with random actions
- [ ] **0.3** Print and log observation/action space shapes, dtypes, and value ranges for both envs
- [ ] **0.4** Run 10 random-action episodes; render a few frames to confirm the simulator looks correct
- [ ] **0.5** Write `src/envs/pickplace_wrapper.py` — a thin Gym-compatible wrapper around robosuite PickPlace
- [ ] **0.6** Write `src/envs/nutassembly_wrapper.py` — same for NutAssembly
- [ ] **0.7** Commit: `feat: environment wrappers and setup verification`

---

## Phase 1 — Single-Object Baseline (SAC + HER)
**Goal:** Prove that SAC + HER can learn a single pick-and-place task. This is the foundational skill everything else builds on.
**Exit criterion:** Agent places one can into one container with >80% success rate on 50 eval episodes.

- [ ] **1.1** Configure SAC + HER in Stable-Baselines3 for single-object PickPlace (`src/agents/sac/train_single.py`)
- [ ] **1.2** Set up W&B run — log episode reward, success rate, and actor/critic losses
- [ ] **1.3** Define hyperparameter config in `configs/sac_single.yaml` (learning rate, buffer size, HER strategy)
- [ ] **1.4** Run training (~500k steps); monitor W&B for convergence
- [ ] **1.5** Evaluate: run 50 episodes with greedy policy, record success rate
- [ ] **1.6** Save best checkpoint to `models/sac_single_best.pt`
- [ ] **1.7** Record a short demo video (`videos/phase1_single_object.mp4`)
- [ ] **1.8** Log milestone in `PROGRESS.md`
- [ ] **1.9** Commit: `feat: SAC+HER single-object baseline`

---

## Phase 2 — Full 4-Object PickPlace (Flat Baseline)
**Goal:** Scale to all 4 objects with flat SAC + HER. Observe where it struggles — this motivates the hierarchical approach.
**Exit criterion:** Agent trained; success rate measured and recorded (even if low). Failure modes documented.

- [ ] **2.1** Update env wrapper to expose all 4 objects and 4 containers
- [ ] **2.2** Adapt reward: +1 per correctly placed object (sum up to +4)
- [ ] **2.3** Train flat SAC + HER for ~1M steps; log to W&B
- [ ] **2.4** Evaluate: 100 episodes, record per-object and full-episode success rates
- [ ] **2.5** Write analysis: where does the flat agent fail? (likely: no sequencing, forgets placed objects)
- [ ] **2.6** Log findings in `PROGRESS.md` — this is the motivation section for Phase 3
- [ ] **2.7** Commit: `feat: flat SAC+HER 4-object baseline`

---

## Phase 3 — Hierarchical Policy (DQN + SAC)
**Goal:** Build the two-level agent. DQN picks which object to target; SAC executes the motion.
**Exit criterion:** Hierarchical agent outperforms flat baseline on full 4-object PickPlace.

- [ ] **3.1** Define sub-task space: 4 objects × 4 containers = 16 discrete sub-tasks (or simplified: 4 "pick object i" actions if container is implicit)
- [ ] **3.2** Implement high-level DQN in `src/agents/dqn/selector.py` — input: env state vector, output: Q-values over sub-tasks
- [ ] **3.3** Implement hierarchical runner in `src/hierarchical/runner.py` — orchestrates DQN → SAC loop per episode
- [ ] **3.4** Define high-level reward: DQN gets +1 when the sub-task SAC completes is successful
- [ ] **3.5** Train hierarchical agent; log DQN sub-task selection distribution to W&B
- [ ] **3.6** Evaluate: 100 episodes, compare success rate vs Phase 2 flat baseline
- [ ] **3.7** Save checkpoint: `models/hierarchical_pickplace_best.pt`
- [ ] **3.8** Record demo video: `videos/phase3_hierarchical_pickplace.mp4`
- [ ] **3.9** Log milestone in `PROGRESS.md`
- [ ] **3.10** Commit: `feat: hierarchical DQN+SAC policy for 4-object PickPlace`

---

## Phase 4 — NutAssembly Curriculum
**Goal:** Extend the system to Task B (nut assembly). Use curriculum: single-nut first, then dual-nut with orientation.
**Exit criterion:** Agent assembles both nuts onto correct pegs with >70% success rate on 100 eval episodes.

- [ ] **4.1** Verify NutAssembly env wrapper handles orientation in the observation (nut quaternion)
- [ ] **4.2** Train SAC + HER on single-nut variant (`configs/sac_nut_single.yaml`)
- [ ] **4.3** Evaluate single-nut: confirm >80% success before moving on
- [ ] **4.4** Extend to dual-nut: add orientation-matching reward component
- [ ] **4.5** Fine-tune from single-nut checkpoint (transfer learning — don't train from scratch)
- [ ] **4.6** Integrate NutAssembly into the hierarchical runner alongside PickPlace
- [ ] **4.7** Evaluate full pipeline: 4-object sort → 2-nut assembly in one episode
- [ ] **4.8** Save checkpoint: `models/hierarchical_full_best.pt`
- [ ] **4.9** Record demo video: `videos/phase4_full_pipeline.mp4`
- [ ] **4.10** Log milestone in `PROGRESS.md`
- [ ] **4.11** Commit: `feat: NutAssembly curriculum and full pipeline integration`

---

## Phase 5 — Ablation Study
**Goal:** Measure the contribution of each component. Quantify what each piece adds.
**Exit criterion:** All 4 configurations evaluated on 200 episodes each; results table complete.

### Configurations to evaluate
| ID | Configuration |
|----|--------------|
| A  | Flat SAC+HER (Phase 2 result) |
| B  | Hierarchical, no curriculum |
| C  | Hierarchical + curriculum |
| D  | Hierarchical + curriculum + domain randomization |

- [ ] **5.1** Write `src/eval/run_eval.py` — loads any checkpoint, runs N episodes, logs success rate / completion time / per-object accuracy
- [ ] **5.2** Evaluate config A (reuse Phase 2 checkpoint) — 200 episodes
- [ ] **5.3** Train config B (hierarchical, curriculum disabled) — log to W&B
- [ ] **5.4** Evaluate config B — 200 episodes
- [ ] **5.5** Evaluate config C (reuse Phase 3/4 checkpoint) — 200 episodes
- [ ] **5.6** Enable domain randomization (`configs/domain_rand.yaml`): randomize object start poses, friction, lighting
- [ ] **5.7** Train config D with domain randomization — log to W&B
- [ ] **5.8** Evaluate config D — 200 episodes
- [ ] **5.9** Fill in the results table in `README.md`
- [ ] **5.10** Commit: `feat: ablation study complete, results table updated`

---

## Phase 6 — Evaluation, Videos & Write-Up
**Goal:** Polish the project for sharing. Clean results, good demos, clear README.
**Exit criterion:** Repo is self-contained, reproducible, and tells a clear story.

- [ ] **6.1** Record final demo videos for all 4 ablation configs
- [ ] **6.2** Add W&B report link to README
- [ ] **6.3** Add architecture diagram to README (ASCII or image)
- [ ] **6.4** Write "Lessons Learned" section in `PROGRESS.md`
- [ ] **6.5** Final EDUCATION.md pass — fill in any missing explanations from Phases 3–5
- [ ] **6.6** Clean up `src/` — remove dead code, add minimal comments where non-obvious
- [ ] **6.7** Test clean install from scratch: `pip install -r requirements.txt` then `python train.py`
- [ ] **6.8** Tag final version: `git tag -a v1.0.0 -m "full pipeline working"`
- [ ] **6.9** Commit: `docs: final write-up and project polish`
- [ ] **6.10** Push and verify GitHub repo is clean

---

## Phase Summary

| Phase | Focus | Key Output |
|-------|-------|-----------|
| 0 | Setup & verification | Env wrappers, obs/action space logged |
| 1 | SAC+HER single object | Baseline checkpoint, >80% success |
| 2 | Flat 4-object baseline | Failure mode analysis, motivation for hierarchy |
| 3 | Hierarchical DQN+SAC | Full PickPlace agent, outperforms flat |
| 4 | NutAssembly curriculum | Full pipeline: sort + assemble |
| 5 | Ablation study | Results table with 4 configs × 200 episodes |
| 6 | Polish & write-up | Clean repo, videos, tagged release |

---

## Current Status

**Active phase:** Phase 0
**Last completed step:** —
**Blockers:** None
