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
        self.bridge_reward_value = 0.1 # Value for forming a bridge
        self.opponent_policy = None

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

    def _get_player_stones(self, player):
        """Returns list of (row,col) coordinates for all stones belonging to player"""
        stones = []
        for r in range(self.hex_game.size):
            for c in range(self.hex_game.size):
                if self.hex_game.board[r][c] == player:
                    stones.append((r, c))
        return stones

    def _calculate_bridge_reward(self, coordinates, player):
        reward = 0
        player_stones = self._get_player_stones(player)  # Pre-compute

        for stone in player_stones:
            if stone == coordinates:
                continue

            common_empty = [
                coord for coord in
                set(self.hex_game._get_adjacent(coordinates)) &
                set(self.hex_game._get_adjacent(stone))
                if self.hex_game.board[coord[0]][coord[1]] == 0
            ]

            if len(common_empty) == 2:
                reward += self.bridge_reward_value
                # break  # Stop after first bridge
        return reward

    def _calculate_center_control(self, coords):
        """Rewards controlling central hexes"""
        center = self.hex_game.size // 2
        distance = max(abs(coords[0] - center), abs(coords[1] - center))

        # Normalized reward based on distance from center
        return max(0, 0.2 * (1 - distance / center))

    def _calculate_chain_reward(self, new_stone, player):
        """Rewards creating longer connected stone groups"""
        visited = set()
        queue = [new_stone]
        chain_length = 0

        while queue:
            current = queue.pop()
            if current in visited:
                continue

            visited.add(current)
            chain_length += 1

            for neighbor in self.hex_game._get_adjacent(current):
                if (self.hex_game.board[neighbor[0]][neighbor[1]] == player and
                        neighbor not in visited):
                    queue.append(neighbor)

        # Scaling reward (capped at 5 stones)
        return 0.1 * min(chain_length, 5)

    def step(self, action):
        coordinates = self.hex_game.scalar_to_coordinates(action)

        # Check if the move is valid
        if coordinates not in self.hex_game.get_action_space():
            # Invalid move, penalize and end episode
            reward = -10 # Large penalty for invalid moves
            terminated = True
            truncated = False # Not used in this env
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        original_player = self.hex_game.player
        
        # Make the move
        self.hex_game.move(coordinates) # board is updated here, self.hex_game.player is flipped

        # initialize rewards
        strategic_reward = 0.0
        final_reward = 0.0
        terminated = False
        truncated = False  # Not used in this env

        # Calculate strategic rewards
        strategic_reward += self._calculate_bridge_reward(coordinates, original_player)
        strategic_reward += self._calculate_center_control(coordinates)
        strategic_reward += self._calculate_chain_reward(coordinates, original_player)
        

        self.hex_game.evaluate() # Check for game end (updates self.hex_game.winner)


        if self.hex_game.winner != 0:
            terminated = True
            final_reward = 10.0 if self.hex_game.winner == original_player else -10.0 # player who made move won or opponent won

        # if there is an opponent policy defined
        elif not terminated and self.opponent_policy is not None and self.hex_game.player == -original_player:
            # Opponent makes a move
            board = self.hex_game.board
            valid_actions = self.hex_game.get_action_space()

            opponent_action_coords = self.opponent_policy(board, valid_actions)
            self.hex_game.move(opponent_action_coords)
            self.hex_game.evaluate()

            # check if the opponent won
            if self.hex_game.winner != 0:
                terminated = True
                final_reward = -10.0 if self.hex_game.winner == -original_player else 10.0 # opponent won or we won (due to illegal move or something)


        reward = final_reward + strategic_reward
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.hex_game.print()

    def close(self):
        pass

    def set_opponent_policy(self, policy_fn):
        self.opponent_policy = policy_fn