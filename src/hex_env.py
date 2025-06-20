import random
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hex_engine import hexPosition

WITH_REWARD_SHAPING = True # set to false if no reward shaping wanted
if WITH_REWARD_SHAPING:
    MAX_REWARD = 2.0
    MOVE_PENALTY_SINGLE_VALUE = 0.0  # should be in good relation to max reward (i.e., if max reward 1, 0.002 is better)
else:
    MAX_REWARD = 1.0
    MOVE_PENALTY_SINGLE_VALUE = 0.0  # should be in good relation to max reward (i.e., if max reward 1, 0.002 is better)

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

    def _calculate_bridge_reward_old(self, coordinates, player):
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

    def _calculate_bridge_reward(self, coordinates, player):
        reward = 0
        player_stones = self._get_player_stones(player)
        bridge_count = 0

        center = self.hex_game.size // 2
        distance_from_center = max(abs(coordinates[0] - center),
                                   abs(coordinates[1] - center))
        position_weight = 1.5 - (distance_from_center / center)

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
                bridge_count += 1

                scaling = math.log(bridge_count + 1.5)

                bridge_value = self.bridge_reward_value * position_weight * scaling

                if self._is_near_opponent_side(stone):
                    bridge_value *= 1.3

                reward += bridge_value

        return min(reward, 1.5)

    def _is_near_opponent_side(self, coords):
        """Check if stone is near opponent's connection side"""
        if self.hex_game.player == 1:  # Player 1 connects top-bottom
            return coords[0] <= 2 or coords[0] >= self.hex_game.size - 3
        else:  # Player 2 connects left-right
            return coords[1] <= 2 or coords[1] >= self.hex_game.size - 3

    def _calculate_center_control(self, coords):
        """Rewards controlling central hexes"""
        center = self.hex_game.size // 2
        distance = max(abs(coords[0] - center), abs(coords[1] - center))

        # Normalized reward based on distance from center
        #print("Center control ", max(0, 0.2 * (1 - distance / center)))
        return max(0, 0.3 * (1 - distance / center)) # max 0.2

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

        # Scaling reward (capped at 5 stones) - max 0.5
        # print("Chain length: {}".format(chain_length))
        return 0.1 * min(chain_length, 5) # max 0.06 * 5

    def get_final_reward(self, original_player):
        if WITH_REWARD_SHAPING:
            base = MAX_REWARD if self.hex_game.winner == original_player else -MAX_REWARD
        else:
            base = MAX_REWARD if self.hex_game.winner == original_player else -MAX_REWARD
        #print("BASE REWARD: ", base)
        #move_penalty = -MOVE_PENALTY_SINGLE_VALUE * self.hex_game.move_count # idea: faster games get better final rewards
        #print("MOVE PENALTY: ", move_penalty)
        return base #+ move_penalty

    def _calculate_strategic_rewards(self, coordinates, player):
        """Weighted rewards"""
        rewards = {
            'bridge': self._calculate_bridge_reward(coordinates, player) * 0.4, # 1.0,
            'center': self._calculate_center_control(coordinates) * 0.3, #1.0,
            'chain': self._calculate_chain_reward(coordinates, player) * 0.4, #0.5,
            'connectivity': self._reward_connectivity(coordinates, player) * 0.4, # 0.5,
            'blocking': self._reward_opponent_blocking(coordinates, player) *0.3, #  1.0
        }
        return sum(rewards.values())

    def _reward_connectivity(self, coords, player):
        """Reward if near victory"""
        if player == 1:  # Player 1 needs vertical connection
            progress = coords[0] / (self.size - 1)  # 0 = top, 1 = bottom
        else:  # player two needs horizontal connection
            progress = coords[1] / (self.size - 1) # 0 = left, 1 = right
        return 0.6 * progress

    def _calculate_opponent_chain_length(self, start_coord, opponent):
        visited = set()
        queue = [start_coord]
        chain_length = 0

        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            chain_length += 1

            for neighbor in self.hex_game._get_adjacent(current):
                if self.hex_game.board[neighbor[0]][neighbor[1]] == opponent and neighbor not in visited:
                    queue.append(neighbor)

        return chain_length

    def _reward_blocking_potential_connection(self, coords, player):
        opponent = -player
        reward = 0

        for empty_coord in self.hex_game._get_adjacent(coords):
            if self.hex_game.board[empty_coord[0]][empty_coord[1]] != 0:
                continue

            opponent_neighbors = [
                n for n in self.hex_game._get_adjacent(empty_coord)
                if self.hex_game.board[n[0]][n[1]] == opponent
            ]
            if len(opponent_neighbors) >= 2:
                reward += 0.3  # Blocking of potentially connecting fields for opponent

        return min(reward, 0.6)

    def _reward_opponent_blocking(self, coords, player):
        opponent = -player
        blocking_large_chains = 0

        for neighbor in self.hex_game._get_adjacent(coords):
            if self.hex_game.board[neighbor[0]][neighbor[1]] == opponent:
                chain_length = self._calculate_opponent_chain_length(neighbor, opponent)
                blocking_large_chains += min(0.1 * chain_length, 0.6)

        blocking_potential = self._reward_blocking_potential_connection(coords, player)

        total_blocking_reward = blocking_large_chains + blocking_potential
        return min(total_blocking_reward, 1.0) # cap at 1.0

    def _get_return_values(self):
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action, current_payer_in_game):
        coordinates = self.hex_game.scalar_to_coordinates(action)
        original_player = current_payer_in_game

        # Check if the move is valid
        if coordinates not in self.hex_game.get_action_space():
            # Invalid move, penalize and end episode
            reward = -MAX_REWARD # Large penalty for invalid moves
            terminated = True
            truncated = False # Not used in this env
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Make the move
        self.hex_game.move(coordinates) # board is updated here, self.hex_game.player is flipped

        # initialize rewards
        final_reward = 0.0
        terminated = False
        truncated = False  # Not used in this env
        #print("------------------------------------------------------------------------------------")
        if WITH_REWARD_SHAPING:
            strategic_reward = self._calculate_strategic_rewards(coordinates, original_player)
        else:
            strategic_reward = 0.0



        self.hex_game.evaluate() # Check for game end (updates self.hex_game.winner)

        if self.hex_game.winner != 0:
            terminated = True
            final_reward = self.get_final_reward(original_player)
            # print("FINAL Reward: ", final_reward)
            reward = final_reward + strategic_reward
            observation, info = self._get_return_values()
            #print("Winner is - check before opponent: ", self.hex_game.winner)
            #print("Original player is ", original_player)
            #reward, observation, info = self._get_return_values(original_player, strategic_reward)
            return observation, reward, terminated, truncated, info

        # print("FINAL Reward: ", final_reward)
        reward = final_reward + strategic_reward
        observation, info = self._get_return_values()
        """print(
         f"[STEP] strategic reward: {strategic_reward}")  # , * {strategic_weight} noise: {little_random_noise_for_exploration}")
        print(f"CUMULATIVE REWARD: {reward}")
        if self.hex_game.winner == original_player:
            print(f"WON GAME WITH REWARD {reward}")
        elif self.hex_game.winner != 0 and self.hex_game.winner != original_player:
            print(f"LOST GAME WITH REWARD {reward}")
        """
        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            self.hex_game.print()

    def close(self):
        pass