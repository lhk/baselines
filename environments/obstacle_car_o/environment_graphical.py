import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
from skimage.io import imread
from skimage.transform import resize

import environments.obstacle_car.params as params
from environments.obstacle_car.car import Car
from np_draw.sprite import Sprite


class Environment_Graphical(gym.Env):
    def __init__(self):

        # fillcolor
        self.fillvalue = 255

        # set up numpy arrays to be drawn to
        self.canvas = np.zeros((*params.screen_size, 3), dtype=np.uint8)

        self.obstacle_layer = np.zeros((*params.screen_size, 3), dtype=np.uint8)
        self.obstacle_mask = np.zeros((*params.screen_size,), dtype=np.bool)

        self.goal_layer = np.zeros((*params.screen_size, 3), dtype=np.uint8)
        self.goal_mask = np.zeros((*params.screen_size,), dtype=np.bool)

        self.background = np.zeros((*params.screen_size, 3), dtype=np.uint8)

        self.car_layer = np.zeros((*params.screen_size, 3), dtype=np.uint8)
        self.car_mask = np.zeros((*params.screen_size,), dtype=np.bool)

        # load images and set up their masks
        car_img_transp = imread("environments/obstacle_car/assets/car.png")
        car_img_transp = np.transpose(car_img_transp, [1, 0, 2])
        car_img_transp = resize(car_img_transp, params.car_size)
        car_img = car_img_transp[:, :, :3]  # cut away alpha
        car_img = (car_img * 255).astype(np.uint8)
        car_mask = (car_img_transp[:, :, 3] > 0).astype(np.bool)

        obstacle_img = np.zeros((*params.obstacle_size, 3), dtype=np.uint8)
        obstacle_img[:, :, 0] = (
                255 * np.sin(np.linspace(0, 2 * np.pi, params.obstacle_size[0])).reshape((-1, 1))).astype(np.uint8)
        goal_img = np.zeros((*params.goal_size, 3), dtype=np.uint8)
        goal_img[:, :, 1] = (255 * np.sin(np.linspace(0, 4 * np.pi, params.goal_size[1]))).astype(np.uint8)

        obstacle_mask = np.ones(params.obstacle_size, dtype=np.bool)
        goal_mask = np.ones(params.goal_size, dtype=np.bool)

        # the position will be overwritten later
        default_pos = np.zeros((2,))
        self.car_sprite = Sprite(car_img, car_mask, default_pos, 0)
        self.obstacle_sprite = Sprite(obstacle_img, obstacle_mask, default_pos, 0)
        self.goal_sprite = Sprite(goal_img, goal_mask, default_pos, 0)

        # car and car_sprite are not the same
        # one is just for graphics, the other is for dynamic movement of the car
        self.car = Car(default_pos, 0, 0, params)

        self.actions = [[0, 0], [0, -1], [0, 1], [1, 0]]
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.steps = 0

        # set up values for dynamics
        self.car_sprite.set_rotation(0)
        self.car.rot = 0
        self.car.speed = 0

        # set up car, obstacle and goal positions
        car_position = np.array([0, 0], dtype=np.float64)
        car_position[0] = self.np_random.uniform(params.car_size[0] / 2, params.screen_size[0] - params.car_size[0] / 2)
        car_position[1] = params.screen_size[1] - params.car_size[1] / 2
        self.car_sprite.set_position(car_position)
        self.car.pos = car_position

        goal_position = np.array([0, 0])
        goal_position[0] = self.np_random.uniform(0, params.screen_size[0] - params.goal_size[0])
        goal_position[1] = params.goal_size[1] / 2
        self.goal_sprite.set_position(goal_position)

        min_dist = (1.5 * self.car_sprite.dim + min(self.goal_sprite.size))

        self.obstacle_positions = []
        for i in range(params.num_obstacles):
            while True:
                obs_x = self.np_random.rand() * params.screen_size[0]
                obs_y = params.screen_size[1] * 1 / 3 * (1 + self.np_random.rand())
                obstacle_position = np.array([obs_x, obs_y])
                # obstacle must be away from car and goal
                car_dist = np.linalg.norm(obstacle_position - self.car.pos)
                goal_dist = np.linalg.norm(obstacle_position - self.goal_sprite.pos)

                if car_dist > min_dist and goal_dist > min_dist:
                    self.obstacle_positions.append(obstacle_position)
                    break

        # render to background
        self.goal_layer[:] = self.fillvalue
        self.goal_mask[:] = False
        self.goal_sprite.render(self.goal_layer, self.goal_mask)

        self.obstacle_layer[:] = self.fillvalue
        self.obstacle_mask[:] = False
        for obstacle_position in self.obstacle_positions:
            self.obstacle_sprite.set_position(obstacle_position)
            self.obstacle_sprite.render(self.obstacle_layer, self.obstacle_mask)

        self.background[:] = self.fillvalue
        self.background[self.obstacle_mask] = self.obstacle_layer[self.obstacle_mask]
        self.background[self.goal_mask] = self.goal_layer[self.goal_mask]

        return self.render()

    def render(self):
        # TODO: after inheriting from gym.Env this is supposed to do something different
        # refactor to gym interface

        # reset canvas and foreground,
        # background is not rerendered
        self.canvas[:] = self.background
        self.car_layer[:] = self.fillvalue
        self.car_mask[:] = False

        # plot the car
        self.car_sprite.render(self.car_layer, self.car_mask)

        # overlay foreground to canvas
        self.canvas[self.car_mask] = self.car_layer[self.car_mask]

        return self.canvas

    def step(self, action):
        assert self.action_space.contains(action)
        # internally the action is not a number, but a combination of acceleration and steering
        action = self.actions[action]
        obs, rew, done = self.make_action(action)
        return obs, rew, done, {}

    def make_action(self, action):
        acceleration, steering_angle = action

        old_dist = np.linalg.norm(self.car.pos - self.goal_sprite.pos)
        self.car.update(acceleration, steering_angle)
        new_dist = np.linalg.norm(self.car.pos - self.goal_sprite.pos)

        # if params.reward_distance != 0,
        # then the environment rewards you for moving closer to the goal
        dist_reward = (old_dist - new_dist) * params.reward_distance

        x, y = self.car.pos

        border_collision = False
        if x > params.screen_size[0]:
            border_collision = True
            self.car.pos[0] = params.screen_size[0]
        elif x < 0:
            border_collision = True
            self.car.pos[0] = 0

        if y > params.screen_size[1]:
            border_collision = True
            self.car.pos[1] = params.screen_size[1]
        elif y < 0:
            border_collision = True
            self.car.pos[1] = 0

        if border_collision:
            self.car.speed = 0

        # sync dynamics and graphics
        self.car_sprite.set_position(self.car.pos)
        self.car_sprite.set_rotation(-self.car.rot)

        # update rendering
        # attention: this has an important side effect:
        # it also updates the occupation masks for foreground and background
        observation = self.render()

        if border_collision and params.stop_on_border_collision:
            return observation, params.reward_collision, True

        reward, collides = self.check_collisions()

        if collides:
            return observation, reward, True

        self.steps += 1
        if self.steps > params.timeout:
            return observation, params.reward_timestep + dist_reward, True

        return observation, params.reward_timestep + dist_reward, False

    def check_collisions(self):
        if np.any(self.car_mask[self.obstacle_mask]):
            return params.reward_collision, True
        if np.any(self.car_mask[self.goal_mask]):
            return params.reward_goal, True

        return 0, False

    def sample_action(self):
        # for atari, the actions are simply numbers
        return self.np_random.choice(self.num_actions)
