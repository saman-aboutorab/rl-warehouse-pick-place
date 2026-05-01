# Project Phases & Steps

Roadmap for `rl-warehouse-pick-place`. Work through phases in order — each one builds on the last.
Update the checkboxes as steps are completed.

---

## Phase 0 — Environment Setup & Verification
**Goal:** Get the simulator running and understand what data the robot sees and produces.
**Exit criterion:** PickPlace environment runs without error; observation and action shapes are logged.

- [x] **0.1** Install dependencies
  **Run:** `pip install -r requirements.txt`
  **Expect:** all packages resolve without error; `pip list` shows robosuite 1.5.x, mujoco 3.x, torch 2.x

- [x] **0.2** Smoke-test robosuite
  **Run:**
  ```python
  import robosuite as suite, numpy as np
  env = suite.make("PickPlace", robots="Panda", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True)
  obs = env.reset(); obs, r, done, info = env.step(np.zeros(env.action_dim)); print("OK", r)
  ```
  **Expect:** prints `OK 0.0` with no Python errors (robosuite WARNINGs about macros are fine)

- [x] **0.3** Log observation/action shapes
  **Run:** inspect each key in `env.reset()` and print `shape`, `dtype`, `min`, `max`
  **Expect:** `robot0_proprio-state` → `[50]`, `object-state` → `[56]` (4-obj) or `[14]` (single), action → `[7]`

- [x] **0.4** Run 10 random-action episodes
  **Run:** loop `env.reset()` + `env.step(random_action)` for 10 episodes
  **Expect:** all episode rewards = `0.0` (sparse reward; random arm never places correctly), lengths = 1000

- [x] **0.5** Write `src/envs/pickplace_wrapper.py`
  **Run:**
  ```bash
  source .venv/bin/activate
  python -c "
  import sys; sys.path.insert(0,'src')
  from envs.pickplace_wrapper import PickPlaceWrapper
  env = PickPlaceWrapper(single_object='Can')
  obs, _ = env.reset()
  print(obs['observation'].shape, obs['achieved_goal'].shape)
  "
  ```
  **Expect:** `(64,) (3,)` — observation flat vector and 3D goal

- [x] **0.6** Commit: `feat: environment wrappers and setup verification`

**Verify phase 0:**
```bash
source .venv/bin/activate && python scripts/verify_phase0.py
```
**Expected:** all sections show correct shapes and live data

---

## Phase 1 — Single-Object Baseline (SAC + HER)
**Goal:** Prove that SAC + HER can learn a single pick-and-place task. This is the foundational skill everything else builds on.
**Exit criterion:** Agent places one Can into its container with >80% success rate on 50 eval episodes.

- [x] **1.1** Write training script `src/agents/sac/train_single.py` — configure SAC + HER in SB3
- [x] **1.2** Write hyperparameter config `configs/sac_single.yaml` — learning rate, buffer size, HER strategy
- [x] **1.3** Set up W&B logging — episode reward, success rate, actor/critic losses
- [ ] **1.4** Run training (~500k steps); monitor W&B for convergence
- [ ] **1.5** Evaluate: run 50 episodes with greedy policy, record success rate
- [ ] **1.6** Save best checkpoint to `models/sac_single_best.zip`
- [ ] **1.7** Record demo video with trained policy
- [ ] **1.8** Log milestone in `PROGRESS.md`
- [ ] **1.9** Commit: `feat: SAC+HER single-object baseline`

**Verify phase 1:**
```bash
source .venv/bin/activate && python scripts/verify_phase1.py
```
**Expected:** shows training curves, success rate >80%, live demo of trained arm

---

## Phase 2 — Full 4-Object PickPlace (Flat Baseline)
**Goal:** Scale to all 4 objects with flat SAC + HER. Observe where it struggles — this motivates the hierarchical approach.
**Exit criterion:** Agent trained; success rate measured and recorded (even if low). Failure modes documented.

- [ ] **2.1** Update env wrapper to expose all 4 objects and 4 containers
- [ ] **2.2** Adapt reward: +1 per correctly placed object (sum up to +4)
- [ ] **2.3** Train flat SAC + HER for ~1M steps; log to W&B
- [ ] **2.4** Evaluate: 100 episodes, record per-object and full-episode success rates
- [ ] **2.5** Document failure modes in `PROGRESS.md` — where and why it gets stuck
- [ ] **2.6** Commit: `feat: flat SAC+HER 4-object baseline`

**Verify phase 2:**
```bash
source .venv/bin/activate && python scripts/verify_phase2.py
```

---

## Phase 3 — Hierarchical Policy (DQN + SAC)
**Goal:** Build the two-level agent. DQN picks which object to target next; SAC executes the motion.
**Exit criterion:** Hierarchical agent outperforms flat baseline on full 4-object PickPlace.

- [ ] **3.1** Implement high-level DQN in `src/agents/dqn/selector.py` — input: which objects remain, output: which to pick next
- [ ] **3.2** Implement hierarchical runner in `src/hierarchical/runner.py` — orchestrates DQN → SAC loop per episode
- [ ] **3.3** Define high-level reward: DQN gets +1 when SAC successfully places the selected object
- [ ] **3.4** Train hierarchical agent; log DQN sub-task selection distribution to W&B
- [ ] **3.5** Evaluate: 100 episodes, compare success rate vs Phase 2 flat baseline
- [ ] **3.6** Save checkpoint: `models/hierarchical_best.zip`
- [ ] **3.7** Record demo video
- [ ] **3.8** Log milestone in `PROGRESS.md`
- [ ] **3.9** Commit: `feat: hierarchical DQN+SAC policy for 4-object PickPlace`

**Verify phase 3:**
```bash
source .venv/bin/activate && python scripts/verify_phase3.py
```

---

## Phase 4 — Ablation Study
**Goal:** Measure the contribution of each component. Quantify what each piece adds.
**Exit criterion:** All configurations evaluated on 200 episodes each; results table complete in README.

### Configurations to evaluate
| ID | Configuration |
|----|--------------|
| A  | Flat SAC+HER — single object (Phase 1) |
| B  | Flat SAC+HER — 4 objects (Phase 2) |
| C  | Hierarchical, no curriculum (Phase 3 variant) |
| D  | Hierarchical + curriculum |
| E  | Hierarchical + curriculum + domain randomisation |

- [ ] **4.1** Write `src/eval/run_eval.py` — loads any checkpoint, runs N episodes, logs success rate / completion time / per-object accuracy
- [ ] **4.2** Evaluate configs A and B (reuse Phase 1 & 2 checkpoints) — 200 episodes each
- [ ] **4.3** Train config C (hierarchical, no curriculum) — log to W&B
- [ ] **4.4** Evaluate configs C and D — 200 episodes each
- [ ] **4.5** Enable domain randomisation (`configs/domain_rand.yaml`): randomise object start poses
- [ ] **4.6** Train and evaluate config E — 200 episodes
- [ ] **4.7** Fill in results table in `README.md`
- [ ] **4.8** Commit: `feat: ablation study complete, results table updated`

**Verify phase 4:**
```bash
source .venv/bin/activate && python scripts/verify_phase4.py
```

---

## Phase 5 — Evaluation, Videos & Write-Up
**Goal:** Polish the project for sharing. Clean results, good demos, clear README.
**Exit criterion:** Repo is self-contained, reproducible, and tells a clear story.

- [ ] **5.1** Record final demo videos for all ablation configs
- [ ] **5.2** Add W&B report link to README
- [ ] **5.3** Add architecture diagram to README
- [ ] **5.4** Write "Lessons Learned" section in `PROGRESS.md`
- [ ] **5.5** Final EDUCATION.md pass — fill in any gaps from Phases 1–4
- [ ] **5.6** Test clean install from scratch: `pip install -r requirements.txt` then train
- [ ] **5.7** Tag final version: `git tag -a v1.0.0 -m "full pipeline working"`
- [ ] **5.8** Commit: `docs: final write-up and project polish`

---

## Phase Summary

| Phase | Focus | Key Output |
|-------|-------|-----------|
| 0 ✓ | Setup & verification | Env wrapper, obs/action space confirmed |
| 1 | SAC+HER single object | Baseline checkpoint, >80% success |
| 2 | Flat 4-object baseline | Failure mode analysis, motivation for hierarchy |
| 3 | Hierarchical DQN+SAC | Full PickPlace agent, outperforms flat |
| 4 | Ablation study | Results table with all configs × 200 episodes |
| 5 | Polish & write-up | Clean repo, videos, tagged release |

---

## Current Status

**Active phase:** Phase 1
**Last completed step:** 0.6 — Phase 0 complete
**Blockers:** None
