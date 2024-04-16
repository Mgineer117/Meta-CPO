# import the objects you want to use
# or you can define specific objects by yourself, just make sure obeying our specification
import random
import numpy as np

from safety_gymnasium.assets.geoms import Circle
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.bases.base_task import BaseTask

def generate_number(min_value, max_value, seed=10, is_int=False):
    random.seed(seed)
    random_number = random.uniform(min_value, max_value)
    if is_int:
        rounded_number = round(random_number)
    else:
        rounded_number = round(random_number, 2)
    return rounded_number

class StcircleLevel0(BaseTask):
    """An agent want to loop around the boundary of circle."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # define some properties

        self.num_steps = 500

        self.agent.placements = [(-0.2, -0.2, 0.2, 0.2)]
        self.agent.keepout = 0
        
        self.lidar_conf.max_dist = 6

        # Reward for circle goal (complicated formula depending on pos and vel)
        self.reward_factor: float = 1e-1

        # add objects into environments
        radius = 1.25
        scale = 0.65
        num_hazards = 5
        #radius = 1.5
        #scale = 0.85
        distance = round(radius*scale, 3)
        num_walls = 2
        #num_hazards = generate_number(4, 8, is_int=True)
        print('radius: ', radius)
        print('distance: ', distance)
        print('num walls: ', num_walls)
        print('num hazards: ', num_hazards)

        self._add_geoms(Circle(radius=radius))
        self._add_geoms(Hazards(num=num_hazards, size=0.12, cost=3, placements=[(-1.4, -1.4, 1.4, 1.4)], is_constrained=True))
        self._add_geoms(Sigwalls(num = num_walls, locate_factor=distance, size=3.5, is_constrained=True))


    def calculate_reward(self):
        # implement your reward function
        # Note: cost calculation is based on objects, so it's automatic
        reward = 0.0
        # Circle environment reward
        agent_pos = self.agent.pos
        agent_vel = self.agent.vel
        x, y, _ = agent_pos
        u, v, _ = agent_vel
        radius = np.sqrt(x**2 + y**2)
        reward += (
            ((-u * y + v * x) / radius)
            / (1 + np.abs(radius - self.circle.radius))  # pylint: disable=no-member
        ) * self.reward_factor
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        # depending on your task
        pass

    def update_world(self):
        # depending on your task
        pass

    @property
    def goal_achieved(self):
        # depending on your task
        return False
