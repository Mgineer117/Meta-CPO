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
"""The vectorized Safety-Gymnasium wrapper."""

from __future__ import annotations

from typing import Iterable

import gymnasium
from gymnasium.vector.vector_env import VectorEnv

from safety_gymnasium.utils.registration import make as safety_make
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv
from safety_gymnasium.vector.sync_vector_env import SafetySyncVectorEnv


__all__ = ['SafetyAsyncVectorEnv', 'SafetySyncVectorEnv', 'VectorEnv', 'make']


def make(
    env_id: str,
    num_envs: int = 1,
    asynchronous: bool = True,
    wrappers: callable | list[callable] | None = None,
    disable_env_checker: bool | None = None,
    **kwargs,
) -> VectorEnv:
    """Create a vectorized environment from multiple copies of an environment, from its id.

    Example::

        >>> import safety_gymnasium
        >>> env = safety_gymnasium.vector.make('SafetyPointGoal1-v0', num_envs=3)
        >>> env.obsevation_space
        Box(-inf, inf, (3, 60), float64)
        >>> env.action_space
        Box(-1.0, 1.0, (3, 2), float64)

    Args:
        env_id: The environment id. This must be a valid ID from the registry.
        num_envs: Number of copies of the environment.
        asynchronous: If `True`, wraps the environments in an :class:`AsyncVectorEnv`
        (which uses `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.
        wrappers: If not ``None``, then apply the wrappers to each internal environment during creation.
        disable_env_checker: If to run the env checker for the first environment only.
        None will default to the environment spec `disable_env_checker` parameter
        (that is by default False), otherwise will run according to this argument (True=not run, False=run)
        **kwargs: Keywords arguments applied during `safety_gymnasium.make`
    """

    def create_env(env_num: int) -> callable:
        """Creates an environment that can enable or disable the environment checker."""
        # if the env_num > 0 then disable the environment checker otherwise use the parameter.
        _disable_env_checker = True if env_num > 0 else disable_env_checker

        def _make_env() -> gymnasium.Env:
            """Make the environment."""
            env = safety_make(
                env_id,
                disable_env_checker=_disable_env_checker,
                **kwargs,
            )
            if wrappers is not None:
                if callable(wrappers):
                    env = wrappers(env)
                elif isinstance(wrappers, Iterable):
                    for wrapper in wrappers:
                        if callable(wrapper):
                            env = wrapper(env)
                        else:
                            raise NotImplementedError
            return env

        return _make_env

    env_fns = [create_env(disable_env_checker or env_num > 0) for env_num in range(num_envs)]
    return SafetyAsyncVectorEnv(env_fns) if asynchronous else SafetySyncVectorEnv(env_fns)
