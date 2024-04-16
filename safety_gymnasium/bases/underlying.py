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
"""Base mujoco task."""

from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import dataclass

import gymnasium
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

import safety_gymnasium
from safety_gymnasium import agents
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.free_geoms import FREE_GEOMS_REGISTER
from safety_gymnasium.assets.geoms import GEOMS_REGISTER
from safety_gymnasium.assets.mocaps import MOCAPS_REGISTER
from safety_gymnasium.bases.base_object import FreeGeom, Geom, Mocap
from safety_gymnasium.utils.common_utils import MujocoException
from safety_gymnasium.utils.keyboard_viewer import KeyboardViewer
from safety_gymnasium.utils.random_generator import RandomGenerator
from safety_gymnasium.world import World


@dataclass
class RenderConf:
    r"""Render options.

    Attributes:
        libels (bool): Whether to render labels.
        lidar_markers (bool): Whether to render lidar markers.
        lidar_radius (float): Radius of the lidar markers.
        lidar_size (float): Size of the lidar markers.
        lidar_offset_init (float): Initial offset of the lidar markers.
        lidar_offset_delta (float): Delta offset of the lidar markers.
    """

    labels: bool = False
    lidar_markers: bool = True
    lidar_radius: float = 0.15
    lidar_size: float = 0.025
    lidar_offset_init: float = 0.5
    lidar_offset_delta: float = 0.06


@dataclass
class PlacementsConf:
    r"""Placement options.

    Attributes:
        placements (dict): Generated during running.
        extents (list): Placement limits (min X, min Y, max X, max Y).
        margin (float): Additional margin added to keepout when placing objects.
    """

    placements = None
    # FIXME: fix mutable default arguments  # pylint: disable=fixme
    extents = (-2, -2, 2, 2)
    margin = 0.0


@dataclass
class SimulationConf:
    r"""Simulation options.

    Note:
        Frameskip is the number of physics simulation steps per environment step and is sampled
        as a binomial distribution.
        For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip).

    Attributes:
        frameskip_binom_n (int): Number of draws trials in binomial distribution (max frameskip).
        frameskip_binom_p (float): Probability of trial return (controls distribution).
    """

    frameskip_binom_n: int = 10
    frameskip_binom_p: float = 1.0


@dataclass
class VisionEnvConf:
    r"""Vision observation parameters.

    Attributes:
        vision_size (tuple): Size (width, height) of vision observation.
    """

    vision_size = (256, 256)


@dataclass
class FloorConf:
    r"""Floor options.

    Attributes:
        type (str): Type of floor.
        size (tuple): Size of floor in environments.
    """

    type: str = 'mat'  # choose from 'mat' and 'village'
    size: tuple = (3.5, 3.5, 0.1)


@dataclass
class WorldInfo:
    r"""World information generated in running.

    Attributes:
        layout (dict): Layout of the world.
        reset_layout (dict): Saved layout of the world after reset.
        world_config_dict (dict): World configuration dictionary.
    """

    layout: dict = None
    reset_layout: dict = None
    world_config_dict: dict = None


class Underlying(abc.ABC):  # pylint: disable=too-many-instance-attributes
    r"""Base class which is in charge of mujoco and underlying process.

    Methods:

    - :meth:`_parse`: Parse the configuration from dictionary.
    - :meth:`_build_agent`: Build the agent instance.
    - :meth:`_add_geoms`: Add geoms into current environment.
    - :meth:`_add_free_geoms`: Add free geoms into current environment.
    - :meth:`_add_mocaps`: Add mocaps into current environment.
    - :meth:`reset`: Reset the environment, it is dependent on :meth:`_build`.
    - :meth:`_build`: Build the mujoco instance of environment from configurations.
    - :meth:`simulation_forward`: Forward the simulation.
    - :meth:`update_layout`: Update the layout dictionary of the world to update states of some objects.
    - :meth:`_set_goal`: Set the goal position in physical simulator.
    - :meth:`_render_lidar`: Render the lidar.
    - :meth:`_render_compass`: Render the compass.
    - :meth:`_render_area`: Render the area.
    - :meth:`_render_sphere`: Render the sphere.
    - :meth:`render`: Render the environment, it may call :meth:`_render_lidar`, :meth:`_render_compass`
      :meth:`_render_area`, :meth:`_render_sphere`.
    - :meth:`_get_viewer`: Get the viewer instance according to render_mode.
    - :meth:`_update_viewer`: Update the viewer when world is updated.
    - :meth:`_obs_lidar`: Get observations from the lidar.
    - :meth:`_obs_compass`: Get observations from the compass.
    - :meth:`_build_placements_dict`: Build the placements dictionary for different types of object.
    - :meth:`_build_world_config`: Build the world configuration, combine separate configurations from
        different types of objects together as world configuration.

    Attributes:

    - :attr:`sim_conf` (SimulationConf): Simulation options.
    - :attr:`placements_conf` (PlacementsConf): Placement options.
    - :attr:`render_conf` (RenderConf): Render options.
    - :attr:`vision_env_conf` (VisionEnvConf): Vision observation parameters.
    - :attr:`floor_conf` (FloorConf): Floor options.
    - :attr:`random_generator` (RandomGenerator): Random generator instance.
    - :attr:`world` (World): World, which is in charge of mujoco.
    - :attr:`world_info` (WorldInfo): World information generated according to environment in running.
    - :attr:`viewer` (Union[KeyboardViewer, RenderContextOffscreen]): Viewer for environment.
    - :attr:`_viewers` (dict): Viewers.
    - :attr:`_geoms` (dict): Geoms which are added into current environment.
    - :attr:`_free_geoms` (dict): FreeGeoms which are added into current environment.
    - :attr:`_mocaps` (dict): Mocaps which are added into current environment.
    - :attr:`agent_name` (str): Name of the agent in current environment.
    - :attr:`observe_vision` (bool): Whether to observe vision from the agent.
    - :attr:`debug` (bool): Whether to enable debug mode, which is pre-config during registration.
    - :attr:`observation_flatten` (bool): Whether to flatten the observation.
    - :attr:`agent` (Agent): Agent instance added into current environment.
    - :attr:`action_noise` (float): Magnitude of independent per-component gaussian action noise.
    - :attr:`model`: mjModel.
    - :attr:`data`: mjData.
    - :attr:`_obstacles` (list): All types of object in current environment.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the engine.

        Args:
            config (dict): Configuration dictionary, used to pre-config some attributes
                according to tasks via :meth:`safety_gymnasium.register`.
        """

        self.sim_conf = SimulationConf()
        self.placements_conf = PlacementsConf()
        self.render_conf = RenderConf()
        self.vision_env_conf = VisionEnvConf()
        self.floor_conf = FloorConf()

        self.random_generator = RandomGenerator()

        self.world = None
        self.world_info = WorldInfo()

        self.viewer = None
        self._viewers = {}

        # Obstacles which are added in environments.
        self._geoms = {}
        self._free_geoms = {}
        self._mocaps = {}

        # something are parsed from pre-defined configs
        self.agent_name = None
        self.observe_vision = False  # Observe vision from the agent
        self.debug = False
        self.observation_flatten = True  # Flatten observation into a vector
        self._parse(config)
        self.agent = None
        self.action_noise: float = (
            0.0  # Magnitude of independent per-component gaussian action noise
        )
        self._build_agent(self.agent_name)

    def _parse(self, config: dict) -> None:
        """Parse a config dict.

        Modify some attributes according to config.
        So that easily adapt to different environment settings.

        Args:
            config (dict): Configuration dictionary.
        """
        for key, value in config.items():
            if '.' in key:
                obj, key = key.split('.')
                assert hasattr(self, obj) and hasattr(getattr(self, obj), key), f'Bad key {key}'
                setattr(getattr(self, obj), key, value)
            else:
                assert hasattr(self, key), f'Bad key {key}'
                setattr(self, key, value)

    def _build_agent(self, agent_name: str) -> None:
        """Build the agent in the world."""
        assert hasattr(agents, agent_name), 'agent not found'
        agent_cls = getattr(agents, agent_name)
        self.agent = agent_cls(random_generator=self.random_generator)

    def _add_geoms(self, *geoms: Geom) -> None:
        """Register geom type objects into environments and set corresponding attributes."""
        for geom in geoms:
            assert (
                type(geom) in GEOMS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._geoms[geom.name] = geom
            setattr(self, geom.name, geom)
            geom.set_agent(self.agent)

    def _add_free_geoms(self, *free_geoms: FreeGeom) -> None:
        """Register FreeGeom type objects into environments and set corresponding attributes."""
        for obj in free_geoms:
            assert (
                type(obj) in FREE_GEOMS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._free_geoms[obj.name] = obj
            setattr(self, obj.name, obj)
            obj.set_agent(self.agent)

    def _add_mocaps(self, *mocaps: Mocap) -> None:
        """Register mocap type objects into environments and set corresponding attributes."""
        for mocap in mocaps:
            assert (
                type(mocap) in MOCAPS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._mocaps[mocap.name] = mocap
            setattr(self, mocap.name, mocap)
            mocap.set_agent(self.agent)

    def reset(self) -> None:
        """Reset the environment."""
        self._build()
        # Save the layout at reset
        self.world_info.reset_layout = deepcopy(self.world_info.layout)

    def _build(self) -> None:
        """Build the mujoco instance of environment from configurations."""
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        # Sample object positions
        self.world_info.layout = self.random_generator.build_layout()

        # Build the underlying physics world
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)

        if self.world is None:
            self.world = World(self.agent, self._obstacles, self.world_info.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)

    def simulation_forward(self, action: np.ndarray) -> None:
        """Take a step in the physics simulation.

        Note:
            - The **step** mentioned above is not the same as the **step** in Mujoco sense.
            - The **step** here is the step in episode sense.
        """
        # Simulate physics forward
        if self.debug:
            self.agent.debug()
        else:
            noise = (
                self.action_noise * self.random_generator.randn(self.agent.body_info.nu)
                if self.action_noise
                else None
            )
            self.agent.apply_action(action, noise)

        exception = False
        for _ in range(
            self.random_generator.binomial(
                self.sim_conf.frameskip_binom_n,
                self.sim_conf.frameskip_binom_p,
            ),
        ):
            try:
                for mocap in self._mocaps.values():
                    mocap.move()
                # pylint: disable-next=no-member
                mujoco.mj_step(self.model, self.data)  # Physics simulation step
            except MujocoException as me:  # pylint: disable=invalid-name
                print('MujocoException', me)
                exception = True
                break
        if exception:
            return exception

        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor readings correct!
        return exception

    def update_layout(self) -> None:
        """Update layout dictionary with new places of objects from Mujoco instance.

        When the objects moves, and if we want to update locations of some objects in environment,
        then the layout dictionary needs to be updated to make sure that we won't wrongly change
        the locations of other objects because we build world according to layout dictionary.
        """
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member
        for k in list(self.world_info.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k:
                continue
            self.world_info.layout[k] = self.data.body(k).xpos[:2].copy()

    def _set_goal(self, pos: np.ndarray) -> None:
        """Set position of goal object in Mujoco instance.

        Note:
            This method is used to make sure the position of goal object in Mujoco instance
            is the same as the position of goal object in layout dictionary or in attributes
            of task instance.
        """
        if pos.shape == (2,):
            self.model.body('goal').pos[:2] = pos[:2]
        elif pos.shape == (3,):
            self.model.body('goal').pos[:3] = pos[:3]
        else:
            raise NotImplementedError

    def _render_lidar(
        self,
        poses: np.ndarray,
        color: np.ndarray,
        offset: float,
        group: int,
    ) -> None:
        """Render the lidar observation."""
        agent_pos = self.agent.pos
        agent_mat = self.agent.mat
        lidar = self._obs_lidar(poses, group)
        for i, sensor in enumerate(lidar):
            if self.lidar_conf.type == 'pseudo':  # pylint: disable=no-member
                i += 0.5  # Offset to center of bin
            theta = 2 * np.pi * i / self.lidar_conf.num_bins  # pylint: disable=no-member
            rad = self.render_conf.lidar_radius
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = agent_pos + np.matmul(binpos, agent_mat.transpose())
            alpha = min(1, sensor + 0.1)
            self.viewer.add_marker(
                pos=pos,
                size=self.render_conf.lidar_size * np.ones(3),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
                rgba=np.array(color) * alpha,
                label='',
            )

    def _render_compass(self, pose: np.ndarray, color: np.ndarray, offset: float) -> None:
        """Render a compass observation."""
        agent_pos = self.agent.pos
        agent_mat = self.agent.mat
        # Truncate the compass to only visualize XY component
        compass = np.concatenate([self._obs_compass(pose)[:2] * 0.15, [offset]])
        pos = agent_pos + np.matmul(compass, agent_mat.transpose())
        self.viewer.add_marker(
            pos=pos,
            size=0.05 * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * 0.5,
            label='',
        )

    # pylint: disable-next=too-many-arguments
    def _render_area(
        self,
        pos: np.ndarray,
        size: float,
        color: np.ndarray,
        label: str = '',
        alpha: float = 0.1,
    ) -> None:
        """Render a radial area in the environment."""
        z_size = min(size, 0.3)
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=[size, size, z_size],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,  # pylint: disable=no-member
            rgba=np.array(color) * alpha,
            label=label if self.render_conf.labels else '',
        )

    # pylint: disable-next=too-many-arguments
    def _render_sphere(
        self,
        pos: np.ndarray,
        size: float,
        color: np.ndarray,
        label: str = '',
        alpha: float = 0.1,
    ) -> None:
        """Render a radial area in the environment."""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=size * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * alpha,
            label=label if self.render_conf.labels else '',
        )

    # pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements
    def render(
        self,
        width: int,
        height: int,
        mode: str,
        camera_id: int | None = None,
        camera_name: str | None = None,
        cost: float | None = None,
    ) -> None:
        """Render the environment to somewhere.

        Note:
            The camera_name parameter can be chosen from:
              - **human**: the camera used for freely moving around and can get input
                from keyboard real time.
              - **vision**: the camera used for vision observation, which is fixed in front of the
                agent's head.
              - **track**: The camera used for tracking the agent.
              - **fixednear**: the camera used for top-down observation.
              - **fixedfar**: the camera used for top-down observation, but is further than **fixednear**.
        """
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height

        if mode in {
            'rgb_array',
            'depth_array',
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    'Both `camera_id` and `camera_name` cannot be specified at the same time.',
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'vision'

            if camera_id is None:
                # pylint: disable-next=no-member
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,  # pylint: disable=no-member
                    camera_name,
                )

        self._get_viewer(mode)

        # Turn all the geom groups on
        self.viewer.vopt.geomgroup[:] = 1

        # Lidar and Compass markers
        if self.render_conf.lidar_markers:
            offset = (
                self.render_conf.lidar_offset_init
            )  # Height offset for successive lidar indicators
            for obstacle in self._obstacles:
                if obstacle.is_lidar_observed:
                    self._render_lidar(obstacle.pos, obstacle.color, offset, obstacle.group)
                if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                    self._render_compass(
                        getattr(self, obstacle.name + '_pos'),
                        obstacle.color,
                        offset,
                    )
                offset += self.render_conf.lidar_offset_delta

        # Add indicator for nonzero cost
        if cost.get('cost_sum', 0) > 0:
            self._render_sphere(self.agent.pos, 0.25, COLOR['red'], alpha=0.5)

        # Draw vision pixels
        if mode in {'rgb_array', 'depth_array'}:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).render(render_mode=mode, camera_id=camera_id)
            self.viewer._markers[:] = []  # pylint: disable=protected-access
            self.viewer._overlays.clear()  # pylint: disable=protected-access
            return data
        if mode == 'human':
            self._get_viewer(mode).render()
            return None
        raise NotImplementedError(f'Render mode {mode} is not implemented.')

    def _get_viewer(
        self,
        mode: str,
    ) -> (
        safety_gymnasium.utils.keyboard_viewer.KeyboardViewer
        | gymnasium.envs.mujoco.mujoco_rendering.RenderContextOffscreen
    ):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = KeyboardViewer(
                    self.model,
                    self.data,
                    self.agent.keyboard_control_callback,
                )
            elif mode in {'rgb_array', 'depth_array'}:
                self.viewer = OffScreenViewer(self.model, self.data)
            else:
                raise AttributeError(f'Unexpected mode: {mode}')

            # self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def _update_viewer(self, model, data) -> None:
        """update the viewer with new model and data"""
        assert self.viewer, 'Call before self.viewer existing.'
        self.viewer.model = model
        self.viewer.data = data

    @abc.abstractmethod
    def _obs_lidar(self, positions: np.ndarray, group: int) -> np.ndarray:
        """Calculate and return a lidar observation.  See sub methods for implementation."""

    @abc.abstractmethod
    def _obs_compass(self, pos: np.ndarray) -> np.ndarray:
        """Return an agent-centric compass observation of a list of positions.

        Compass is a normalized (unit-length) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

    @abc.abstractmethod
    def _build_placements_dict(self) -> dict:
        """Build a dict of placements.  Happens only once."""

    @abc.abstractmethod
    def _build_world_config(self, layout: dict) -> dict:
        """Create a world_config from our own config."""

    @property
    def model(self):
        """Helper to get the world's model instance."""
        return self.world.model

    @property
    def data(self):
        """Helper to get the world's simulation data instance."""
        return self.world.data

    @property
    def _obstacles(self) -> list[Geom | FreeGeom | Mocap]:
        """Get the obstacles in the task.

        Combine all types of object in current environment together into single list
        in order to easily iterate them.
        """
        return (
            list(self._geoms.values())
            + list(self._free_geoms.values())
            + list(self._mocaps.values())
        )
