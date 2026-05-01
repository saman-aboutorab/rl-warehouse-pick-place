"""
Watch the robot in MuJoCo.

Runs one episode with random actions and:
  1. Opens a live viewer window so you can watch in real time
  2. Also saves a timestamped video to videos/ so every run is kept

Usage:
    source .venv/bin/activate
    python scripts/view_robot.py                    # PickPlace, random actions
    python scripts/view_robot.py --env NutAssembly  # NutAssembly instead
    python scripts/view_robot.py --steps 300        # shorter episode
    python scripts/view_robot.py --no-viewer        # offscreen only, just save video

Controls (live viewer window):
    ESC / Q   close the window
"""

import argparse
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--env",       default="PickPlace", choices=["PickPlace"])
parser.add_argument("--steps",     type=int, default=500)
parser.add_argument("--no-viewer", action="store_true", help="skip live window, just save video")
args = parser.parse_args()

import robosuite as suite
import cv2

os.makedirs("videos", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"videos/{args.env.lower()}_random_{timestamp}.mp4"

print(f"\nEnvironment : {args.env}")
print(f"Steps       : {args.steps}")
print(f"Video saved : {video_path}")
if not args.no_viewer:
    print(f"Live viewer : opening window — press ESC or Q to close\n")
else:
    print(f"Live viewer : disabled (--no-viewer)\n")

# ── Build env with both live renderer and offscreen camera ──────────────────
env = suite.make(
    args.env,
    robots="Panda",
    has_renderer=not args.no_viewer,           # live window
    has_offscreen_renderer=True,               # for video recording
    use_camera_obs=True,                       # include camera frames in obs
    use_object_obs=True,
    camera_names=["agentview", "frontview"],   # two camera angles
    camera_heights=512,
    camera_widths=512,
    render_camera="frontview",                 # which camera the live window shows
    horizon=args.steps,
    control_freq=20,
)

lo, hi = env.action_spec

# ── Video writer (cv2) ───────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# side-by-side: agentview (above the arm) | frontview (from the front)
out = cv2.VideoWriter(video_path, fourcc, 20.0, (1024, 512))

# ── Run one episode ──────────────────────────────────────────────────────────
obs = env.reset()
print(f"Episode started — robot reset to home position")
print(f"{'step':>5s}  {'reward':>7s}  {'note'}")
print(f"{'─'*5}  {'─'*7}  {'─'*40}")

total_reward = 0
for step in range(args.steps):

    # Random action — the robot flails around with no goal
    action = np.random.uniform(lo, hi)
    obs, reward, done, info = env.step(action)
    total_reward += reward

    # ── Live viewer ──
    if not args.no_viewer:
        env.render()

    # ── Grab frames from both cameras and write to video ──
    # robosuite returns image as (H, W, 3) RGB, origin bottom-left → flip vertically
    agent_frame = obs["agentview_image"][::-1]    # flip: top = sky
    front_frame  = obs["frontview_image"][::-1]

    # Convert RGB → BGR for cv2, stack side by side
    frame = np.concatenate([
        cv2.cvtColor(agent_frame, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(front_frame,  cv2.COLOR_RGB2BGR),
    ], axis=1)   # shape: (512, 1024, 3)

    # Burn step number and reward into the frame
    cv2.putText(frame, f"step {step+1:04d}  reward {total_reward:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"agentview (top-down)", (10, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"frontview (side)", (522, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    out.write(frame)

    # Print progress every 100 steps
    if (step + 1) % 100 == 0 or reward > 0:
        note = "  ← REWARD!" if reward > 0 else ""
        print(f"{step+1:>5d}  {total_reward:>7.1f}{note}")

    if done:
        print(f"\nEpisode ended early at step {step+1}")
        break

out.release()
env.close()

print(f"\n{'─'*50}")
print(f"Episode complete")
print(f"  Steps run    : {step + 1}")
print(f"  Total reward : {total_reward:.1f}  (random arm almost never scores)")
print(f"  Video saved  : {video_path}")
print(f"\nWhat you are seeing in the video:")
print(f"  LEFT  (agentview)  — camera above and behind the robot looking down")
print(f"  RIGHT (frontview)  — camera in front looking at the scene")
print(f"  The arm flails with random joint velocities — reward stays 0")
print(f"  After Phase 1 training, swap in a trained policy and watch it place objects")
