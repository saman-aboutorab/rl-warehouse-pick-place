"""
Gym-compatible wrapper around robosuite NutAssembly.

Flattens the obs dict and exposes achieved_goal / desired_goal for HER.

Observation layout (all float32):
  proprio-state  [50]  joint pos/vel/acc, eef pos/quat, gripper
  object-state   [28]  pos+quat+rel-pos+rel-quat for 2 nuts
  Total obs      [78]

Goal:
  achieved_goal  [3]   current XYZ of the target nut
  desired_goal   [3]   XYZ of the matching peg

Action: [7] joint velocity commands in [-1, 1]
"""

import numpy as np
import gymnasium as gym
import robosuite as suite


NUTS = ["SquareNut", "RoundNut"]
# Maps nut name to peg name inside robosuite
NUT_TO_PEG = {"SquareNut": "SquarePeg", "RoundNut": "RoundPeg"}


class NutAssemblyWrapper(gym.Env):
    """
    Single-nut or dual-nut NutAssembly wrapped for Stable-Baselines3 + HER.

    Args:
        single_nut: if a string ("SquareNut" or "RoundNut"), only that nut
                    is active. If None, both nuts are active (dual-nut mode).
        horizon: max steps per episode.
    """

    def __init__(self, single_nut=None, horizon=500):
        super().__init__()
        self.single_nut = single_nut
        self.horizon = horizon

        self._env = suite.make(
            "NutAssembly",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            horizon=horizon,
            single_object_mode=2 if single_nut else 0,
            nut_type=single_nut.replace("Nut", "").lower() if single_nut else None,
            reward_shaping=False,  # sparse for HER
        )

        lo, hi = self._env.action_spec
        self.action_space = gym.spaces.Box(
            low=lo.astype(np.float32),
            high=hi.astype(np.float32),
            dtype=np.float32,
        )

        # Detect actual dims from a live reset (single_nut mode shrinks object-state)
        _raw = self._env.reset()
        obs_dim = (len(_raw["robot0_proprio-state"]) + len(_raw["object-state"]))
        goal_dim = 3

        self.observation_space = gym.spaces.Dict({
            "observation":   gym.spaces.Box(-np.inf, np.inf, (obs_dim,),  np.float32),
            "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
            "desired_goal":  gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
        })

        self._step = 0

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._step = 0
        raw = self._env.reset()
        obs_dict = self._make_obs(raw)
        return obs_dict, {}

    def step(self, action):
        raw, reward, done, info = self._env.step(action.astype(np.float64))
        self._step += 1
        truncated = self._step >= self.horizon
        obs_dict = self._make_obs(raw)
        return obs_dict, float(reward), bool(done), truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # +1 if nut centre is within 2 cm of peg (tighter than PickPlace)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < 0.02).astype(np.float32)

    def render(self):
        pass

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self, raw):
        proprio = raw["robot0_proprio-state"].astype(np.float32)  # [50]
        objects = raw["object-state"].astype(np.float32)          # [28]
        observation = np.concatenate([proprio, objects])           # [78]

        target = self.single_nut if self.single_nut else "SquareNut"
        achieved_goal = raw[f"{target}_pos"].astype(np.float32)   # [3]
        desired_goal  = self._get_peg_pos(target)                  # [3]

        return {
            "observation":   observation,
            "achieved_goal": achieved_goal,
            "desired_goal":  desired_goal,
        }

    def _get_peg_pos(self, nut_name):
        peg_name = NUT_TO_PEG[nut_name]
        try:
            # robosuite exposes peg body positions via sim
            peg_id = self._env.sim.model.body_name2id(peg_name)
            pos = self._env.sim.data.body_xpos[peg_id].astype(np.float32)
        except Exception:
            fallback = {
                "SquareNut": np.array([0.0,  0.1, 0.82], dtype=np.float32),
                "RoundNut":  np.array([0.0, -0.1, 0.82], dtype=np.float32),
            }
            pos = fallback[nut_name]
        return pos
