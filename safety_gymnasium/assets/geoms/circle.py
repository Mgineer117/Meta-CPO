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
"""Circle."""

from dataclasses import dataclass

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom


@dataclass
class Circle(Geom):  # pylint: disable=too-many-instance-attributes
    """CircleTask specific."""

    name: str = 'circle'
    radius: float = 1.5
    placements: list = None
    locations: tuple = ((0, 0),)
    keepout: float = 0.0

    color: np.array = COLOR['circle']
    group: np.array = GROUP['circle']
    is_lidar_observed: bool = True
    is_constrained: bool = False

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        return {
            'name': 'circle',
            'size': np.array([self.radius, 1e-2]),
            'pos': np.r_[xy_pos, 1e-2],
            'rot': 0,
            'type': 'cylinder',
            'contype': 0,
            'conaffinity': 0,
            'group': GROUP['circle'],
            'rgba': COLOR['circle'] * [1, 1, 1, 0.1],  # transparent
        }

    @property
    def pos(self):
        """Helper to get circle position from layout."""
        return [0, 0, 0]
