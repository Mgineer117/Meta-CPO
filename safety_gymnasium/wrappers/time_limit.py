# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for limiting the time steps of an environment."""


from gymnasium.wrappers.time_limit import TimeLimit


class SafeTimeLimit(TimeLimit):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself,
    this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal
    that originates from the underlying environment as part of the MDP.

    Example:
       >>> from gymnasium.envs.classic_control import CartPoleEnv
       >>> from gymnasium.wrappers import TimeLimit
       >>> env = CartPoleEnv()
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, cost, terminated, truncated, info
