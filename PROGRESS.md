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
2. Verify robosuite setup: run PickPlace and NutAssembly with random actions
3. Understand observation/action spaces — log shapes and ranges
4. Begin SAC + HER training on single-object PickPlace

---
