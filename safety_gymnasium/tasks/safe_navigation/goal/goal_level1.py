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
"""Goal level 0."""

from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.bases.base_task import BaseTask

import random
def generate_number(min_value, max_value, is_int=False):
    random_number = random.uniform(min_value, max_value)
    if is_int:
        rounded_number = round(random_number)
    else:
        rounded_number = round(random_number, 2)
    return rounded_number

class GoalLevel1(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        num_hazards = generate_number(14, 22, is_int=True)

        self._add_geoms(Goal(keepout=0.3))
        self._add_geoms(Hazards(num=num_hazards, keepout=0.18))
        #self._add_mocaps(Gremlins(num=num_gremlins, travel=travel_dist, dist_cost=cost_distance))

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size
