"""
Gym-compatible wrapper around robosuite PickPlace.

Flattens the obs dict into a single vector and exposes the
achieved_goal / desired_goal keys that HER needs.

Observation layout (all float32):
  proprio-state  [50]  joint pos/vel/acc, eef pos/quat, gripper
  object-state   [56]  pos+quat+rel-pos+rel-quat for 4 objects
  Total obs      [106]

Goal:
  achieved_goal  [3]   current XYZ of the target object
  desired_goal   [3]   XYZ of the matching container

Action: [7] joint velocity commands in [-1, 1]
"""

import numpy as np
import gymnasium as gym
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler


OBJECTS = ["Milk", "Bread", "Cereal", "Can"]
# Maps each object name to the bin index robosuite uses internally
OBJECT_TO_BIN = {"Milk": 0, "Bread": 1, "Cereal": 2, "Can": 3}


class PickPlaceWrapper(gym.Env):
    """
    Single-task or multi-task PickPlace wrapped for Stable-Baselines3 + HER.

    Args:
        single_object: if a string (e.g. "Can"), only that object appears and
                       the goal is its matching container. If None, all 4 objects
                       are active (multi-task mode).
        horizon: max steps per episode.
    """

    def __init__(self, single_object=None, horizon=500):
        super().__init__()
        self.single_object = single_object
        self.horizon = horizon

        # Only spawn the requested object when in single-object mode
        object_types = [single_object] if single_object else None

        self._env = suite.make(
            "PickPlace",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            horizon=horizon,
            single_object_mode=2 if single_object else 0,
            object_type=single_object.lower() if single_object else None,
            reward_shaping=False,  # keep reward sparse for HER
        )

        lo, hi = self._env.action_spec
        self.action_space = gym.spaces.Box(
            low=lo.astype(np.float32),
            high=hi.astype(np.float32),
            dtype=np.float32,
        )

        # Detect actual dims from a live reset (single_object_mode shrinks object-state)
        _raw = self._env.reset()
        obs_dim = (len(_raw["robot0_proprio-state"]) + len(_raw["object-state"]))
        goal_dim = 3

        self.observation_space = gym.spaces.Dict({
            "observation":    gym.spaces.Box(-np.inf, np.inf, (obs_dim,),  np.float32),
            "achieved_goal":  gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
            "desired_goal":   gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
        })

        self._step = 0
        self._raw_obs = None

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._step = 0
        raw = self._env.reset()
        self._raw_obs = raw
        obs_dict = self._make_obs(raw)
        return obs_dict, {}

    def step(self, action):
        raw, reward, done, info = self._env.step(action.astype(np.float64))
        self._raw_obs = raw
        self._step += 1
        truncated = self._step >= self.horizon
        obs_dict = self._make_obs(raw)
        return obs_dict, float(reward), bool(done), truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # +1 if object is within 5 cm of the goal (container centre)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < 0.05).astype(np.float32)

    def render(self):
        pass

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self, raw):
        proprio = raw["robot0_proprio-state"].astype(np.float32)  # [50]
        objects = raw["object-state"].astype(np.float32)          # [56]
        observation = np.concatenate([proprio, objects])           # [106]

        # achieved_goal: position of the active object
        target = self.single_object if self.single_object else "Can"
        achieved_goal = raw[f"{target}_pos"].astype(np.float32)   # [3]

        # desired_goal: container position stored inside the raw env
        desired_goal = self._get_container_pos(target)             # [3]

        return {
            "observation":   observation,
            "achieved_goal": achieved_goal,
            "desired_goal":  desired_goal,
        }

    def _get_container_pos(self, object_name):
        bin_id = OBJECT_TO_BIN[object_name]
        # robosuite stores bin positions as target_bin_placements
        try:
            pos = self._env.target_bin_placements[bin_id][:3].astype(np.float32)
        except Exception:
            # fallback: fixed approximate positions if attribute not accessible
            fallback = {
                "Milk":   np.array([ 0.11,  0.28, 0.82], dtype=np.float32),
                "Bread":  np.array([ 0.11, -0.28, 0.82], dtype=np.float32),
                "Cereal": np.array([-0.11,  0.28, 0.82], dtype=np.float32),
                "Can":    np.array([-0.11, -0.28, 0.82], dtype=np.float32),
            }
            pos = fallback[object_name]
        return pos
