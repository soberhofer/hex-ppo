import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hex_engine import hexPosition

class HexEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, size=7, render_mode=None):
        self.size = size
        self.hex_game = hexPosition(size=self.size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.size, self.size), dtype=int)
        self.action_space = spaces.Discrete(self.size * self.size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return np.array(self.hex_game.board)

    def _get_info(self):
        return {"valid_actions": [self.hex_game.coordinate_to_scalar(a) for a in self.hex_game.get_action_space()]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hex_game.reset()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        coordinates = self.hex_game.scalar_to_coordinates(action)
        
        # Check if the move is valid
        if coordinates not in self.hex_game.get_action_space():
            # Invalid move, penalize and end episode
            reward = -10 # Large penalty for invalid moves
            terminated = True
            truncated = False
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        original_player = self.hex_game.player
        self.hex_game.move(coordinates)
        self.hex_game.evaluate()

        reward = 0
        terminated = False
        truncated = False

        if self.hex_game.winner != 0:
            terminated = True
            if self.hex_game.winner == original_player:
                reward = 1 # Current player won
            else:
                reward = -1 # Current player lost

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.hex_game.print()

    def close(self):
        pass
