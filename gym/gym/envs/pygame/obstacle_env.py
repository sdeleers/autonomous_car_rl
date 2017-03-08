"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from ple import PLE
import pygame

logger = logging.getLogger(__name__)

def process_state(state):
    return state.ravel()

    # import scipy.io
    # l = state.ravel()[:-1]
    # ll = np.reshape(l, (64, 64, 4))
    # img0 = np.fliplr(np.rot90(np.repeat(np.expand_dims(ll[:, :, 0], 2), 3, 2), 2))
    # img1 = np.fliplr(np.rot90(np.repeat(np.expand_dims(ll[:, :, 1], 2), 3, 2), 2))
    # img2 = np.fliplr(np.rot90(np.repeat(np.expand_dims(ll[:, :, 2], 2), 3, 2), 2))
    # img3 = np.fliplr(np.rot90(np.repeat(np.expand_dims(ll[:, :, 3], 2), 3, 2), 2))
    # scipy.io.savemat('img_buffer.mat', dict(img0=img0, img1=img1, img2=img2, img3=img3))

class ObstacleEnv(gym.Env):

    def __init__(self, game_name):
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()
        self.game_state = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state, add_noop_action=False)
        self.game_state.init()
        self._action_set = np.sort(self.game_state.getActionSet())
        self.action_space = spaces.Discrete(len(self._action_set))
        # self.observation_space = spaces.Box(np.array([0, 0, 0]), np.inf*np.ones(3))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.game_state.getGameStateDims())
        self.viewer = None
        self.metadata = {
            'render.modes': ['rgb_array'],
            'video.frames_per_second' : 30
        }

    def _get_image(self):
        # return self.game_state.getScreenRGB()
        return np.rot90(self.game_state.getScreenRGB())

    def _step(self, action):
        reward = self.game_state.act(self._action_set[action])
        done = self.game_state.game_over()
        obs = self.game_state.getGameState()
        return obs, reward, done, {}

    def _reset(self):
        self.game_state.reset_game()
        state = self.game_state.getGameState()
        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

