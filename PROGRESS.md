# PROGRESS.md — Error & Milestone Journal

Tracks what went wrong, how it was fixed, and what milestones were reached.
Used for: debugging history, SR&ED documentation, onboarding future contributors.

---

## [2026-05-01] Milestone: Project scaffolded

**What works:**
- README written with full architecture description, algorithm table, ablation plan, results table template, and references
- CLAUDE.md filled in with project goal, stack, and permissions
- Full directory structure created (`src/envs`, `src/agents/sac`, `src/agents/dqn`, `src/hierarchical`, `src/curriculum`, `src/eval`, `configs`, `data`, `models`, `notebooks`, `scripts`, `videos`)
- `requirements.txt` created with core dependencies (robosuite, MuJoCo, PyTorch, SB3, W&B)
- `.gitignore` configured to exclude large files (model weights, videos, W&B logs)
- `EDUCATION.md` created with explanations of SAC, HER, DQN, Hierarchical RL, Curriculum Learning, and observation/action space details

**Git tag:** (not yet tagged — no code to run)

**Next steps:**
1. Install dependencies: `pip install -r requirements.txt`
2. Verify robosuite setup: run PickPlace with random actions
3. Understand observation/action spaces — log shapes and ranges
4. Begin SAC + HER training on single-object PickPlace

---

## [2026-05-01] Milestone: PHASES.md created — roadmap locked in

**What works:**
- Full 5-phase roadmap written in `PHASES.md` covering: env setup, single-object baseline, flat multi-object, hierarchical DQN+SAC, ablation study, and final polish
- Each phase has a clear exit criterion and numbered checklist steps
- Phase summary table added for quick orientation

**Git tag:** (not yet — waiting for first runnable code)

**Next steps:** Begin Phase 0 — install dependencies and smoke-test both robosuite environments

---

## [2026-05-01] Milestone: Phase 1 steps 1.1–1.3 — SAC+HER training script written and verified

**What works:**
- `src/agents/sac/train_single.py` — full SAC+HER training loop using SB3; loads hyperparams from YAML
- `configs/sac_single.yaml` — all hyperparameters in one file (LR, buffer, HER k, eval freq, W&B config)
- W&B logging wired in: episode reward, success rate, eval reward logged per 10 episodes / 20k steps
- Smoke-tested: 200-step run builds cleanly, runs on GPU (cuda), saves final checkpoint
- Architecture verified: actor [70→256→256→14], critics [77→256→256→1], 87k + 172k params, device=cuda

**Next step:** Run full training (step 1.4):
```bash
python src/agents/sac/train_single.py --steps 500000
```

**Reproduce setup:**
```bash
source .venv/bin/activate && python scripts/verify_phase1.py
```
**Expected:** architecture and live HER relabeling demo print correctly

---

## [2026-05-01] Milestone: Phase 0 complete — environments verified and wrapped

**What works:**
- All dependencies installed in `.venv` (torch 2.11 + CUDA 13, robosuite 1.5.2, MuJoCo 3.8, SB3 2.8, W&B 0.26)
- PickPlace env smoke-tested: reset + step OK
- Observation and action spaces logged — key facts:
  - Action space is `[7]` (gripper embedded in composite controller)
  - proprio-state: `[50]`, object-state: `[56]` (4-obj) / `[14]` (single mode)
  - Episode horizon: 1000 steps
- 10 random-action episodes run: reward=0.0 for all — confirms sparse reward and motivates HER
- `src/envs/pickplace_wrapper.py` written — Gym Dict wrapper with achieved_goal/desired_goal
- Wrapper tested: obs shape `[64]` in single-object mode, step/reset OK

**Git tag:** (waiting for first training run)

**Reproduce:**
```bash
source .venv/bin/activate && python scripts/verify_phase0.py
```
**Expected:** all checks print `PASS`

**Next steps:** Phase 1 — configure SAC + HER in SB3 for single-object PickPlace (Can → container)

---
