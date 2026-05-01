## Project Context

<!-- ✏️ Fill this in — Claude Code uses it to make better decisions -->
- **Project name:** rl-warehouse-pick-place
- **Goal:** Train a 7-DoF Panda robot arm via hierarchical RL (SAC + HER + DQN) to sort 4 objects into correct bins — entirely through trial and error, no programmed motions
- **Stack:** Python, robosuite, MuJoCo, PyTorch, Stable-Baselines3, Weights & Biases
- **Repo:** Git-tracked — treat file sizes and history with care

---

## Permissions

- ✅ You have full permission to **read, write, create, and delete files**
- ✅ You may **run shell commands, scripts, and background processes**
- ✅ You may **install packages** (pip, apt) as needed
- ✅ No need to ask for confirmation on routine file operations — just do it

---

## After Every Task — Always Do This

After completing any non-trivial task, you must:

1. **Explain what you did** — a short plain-English summary
2. **Highlight the important parts** — key functions, design decisions, gotchas
3. **Describe how it works** — use a simple example if possible
4. **Note data formats and dimensions** — shapes, types, file formats involved
5. **Describe the architecture** — how components connect (system or data flow)
6. **Write a verify script and run it** — after every phase or significant task, create or update `scripts/verify_phaseN.py`. It must:
   - **Show actual data** — print real values, shapes, tensor contents, rewards, so the user can see what the system is doing, not just whether it works
   - **Walk through a live example** — step through the process (reset → observe → act → reward) and print what happens at each stage
   - **Explain the numbers** — label every printed value so the user knows what they are looking at (e.g. "Can position: [0.09, -0.27, 0.86] — this is where the can spawned in the bin")
   - **Flag problems clearly** — if something is broken, print what went wrong and what to expect instead
   - Be runnable with a single command: `python scripts/verify_phaseN.py`
   - Be run immediately and the output shown in chat before declaring the task done

Then write that explanation into **`EDUCATION.md`** (see below).

Also post a **brief summary in the chat** — 2–4 bullet points covering what was done, what files were created/changed, and what comes next. Include a markdown link to each file or section touched so the user can jump straight to it.

## PHASES.md and PROGRESS.md — How-to-Run Instructions

Whenever a phase or milestone is completed, update the relevant file with the single verify command:

- **`PHASES.md`** — at the bottom of each completed phase block:
  ```
  **Verify:** `source .venv/bin/activate && python scripts/verify_phaseN.py`
  ```
- **`PROGRESS.md`** — under each milestone entry:
  ```
  **Reproduce:** `source .venv/bin/activate && python scripts/verify_phaseN.py`
  **Expected:** all checks print PASS
  ```

---

## EDUCATION.md — Tutorial Log

- File location: `EDUCATION.md` at project root
- **Purpose:** Running tutorial document, written for someone learning the stack
- **Style:** Simple words, no assumed knowledge, explain acronyms
- **Include:**
  - What the code does and why
  - Simple worked example
  - Data formats and tensor/array dimensions (e.g., `[B, C, H, W]`)
  - System and data architecture diagrams (use ASCII or Mermaid)
  - How components interact

**Example entry format:**

```markdown
## [YYYY-MM-DD] Feature: OAK-D Frame Capture

### What it does
Captures RGB frames from the OAK-D Lite camera at ~28fps over USB 2.0.

### How it works
The DepthAI pipeline sends frames to a host queue. We poll the queue in a loop
and convert each frame (numpy array) to a PyTorch tensor for inference.

### Data format
- Raw frame: numpy array, shape `(H, W, 3)`, dtype `uint8`, values 0–255
- After transform: torch tensor, shape `(1, 3, 224, 224)`, dtype `float32`, normalized

### Architecture
Camera → DepthAI Pipeline → Host Queue → numpy → torch.Tensor → Model

### Simple example
frame = queue.get().getCvFrame()   # shape: (480, 640, 3)
tensor = transform(frame)          # shape: (1, 3, 224, 224)
```

---

## PROGRESS.md — Error & Milestone Journal

- File location: `PROGRESS.md` at project root
- **Purpose:** Track what went wrong, how it was fixed, and what milestones were reached
- **Use for:** Debugging history, SR&ED documentation, onboarding future contributors

**Entry format:**

```markdown
## [YYYY-MM-DD] Error: <short title>

**Symptom:** What happened / error message
**Root cause:** Why it happened
**Fix:** What solved it
**Lesson:** What to watch for next time

---

## [YYYY-MM-DD] Milestone: v0.1.0 — Camera pipeline working

**What works:** Live frame capture at 28fps
**Git tag:** `v0.1.0`
**Notes:** USB 2.0 bandwidth is the bottleneck; USB 3.0 would allow higher res
```

---

## POC Mindset — Keep It Simple

- **Prefer working over perfect** — a 10-line script that runs beats a framework that doesn't
- **Avoid premature abstraction** — no base classes or plugin systems until needed
- **One file per concept** — don't split things until they're too big to read
- **Fail loudly** — use asserts and clear error messages, not silent fallbacks
- **No magic** — if it's not obvious what a line does, add a comment

---

## Project Structure — Keep It Clean

```
rl-warehouse-pick-place/
├── CLAUDE.md          ← this file (Claude Code reads it)
├── EDUCATION.md       ← tutorial explanations (auto-updated)
├── PROGRESS.md        ← error log + milestones (auto-updated)
├── README.md          ← human-facing project overview
├── data/              ← raw data (add to .gitignore if large)
├── models/            ← saved checkpoints (.gitignore large files)
├── src/               ← source code
│   ├── capture/       ← camera capture scripts
│   ├── train/         ← training scripts
│   └── inference/     ← inference / Re-ID pipeline
├── notebooks/         ← exploratory Jupyter notebooks
├── scripts/           ← one-off utility scripts
├── requirements.txt
└── .gitignore
```

---

## Git Hygiene

- **Never commit large files** — model weights, datasets, raw video go in `.gitignore`
- **Use Git LFS** if you must track binary files (e.g., small reference images)
- **Commit messages:** `type: short description` — e.g., `feat: add OAK-D capture loop`
- **Tag version milestones:** `git tag -a v0.1.0 -m "camera pipeline working"`
- **Common types:** `feat`, `fix`, `docs`, `refactor`, `chore`

### Suggested .gitignore additions
```
data/
models/*.pth
models/*.pt
*.egg-info/
__pycache__/
.venv/
wandb/
runs/
*.mp4
*.avi
```

---

## Commit Types Quick Reference

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | EDUCATION.md, PROGRESS.md, README updates |
| `refactor` | Code restructure, no behavior change |
| `chore` | Dependency updates, config changes |
| `milestone` | Tag only — marks a working version |
