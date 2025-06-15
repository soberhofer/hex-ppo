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
        """Calculates bridge reward with strategic considerations:
        - Higher reward for bridges in central/important areas
        - Diminishing returns for multiple bridges
        - Penalty for bridges in already secure areas
        """
        reward = 0
        player_stones = self._get_player_stones(player)
        bridge_count = 0
        max_bridges_per_stone = 2  # Avoid over-rewarding many bridges

        # Strategic value based on position (center is more valuable)
        center = self.hex_game.size // 2
        distance_from_center = max(abs(coordinates[0] - center),
                                   abs(coordinates[1] - center))
        position_weight = 1.0 - (distance_from_center / center)

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
                # Base bridge value with positional importance
                bridge_value = self.bridge_reward_value * position_weight

                # Diminishing returns for multiple bridges
                bridge_count += 1
                if bridge_count > max_bridges_per_stone:
                    bridge_value *= 0.7  # Reduce value after 2 bridges

                # Additional bonus if bridge connects to a stone near opponent's side
                if self._is_near_opponent_side(stone):
                    bridge_value *= 1.3

                reward += bridge_value

        return reward

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

        # Scaling reward (capped at 5 stones) - max 0.5
        # print("Chain length: {}".format(chain_length))
        return 0.1 * min(chain_length, 5)

    def get_final_reward(self, original_player):
        return 3.0 if self.hex_game.winner == original_player else -3.0

    def step(self, action):
        coordinates = self.hex_game.scalar_to_coordinates(action)

        # Check if the move is valid
        if coordinates not in self.hex_game.get_action_space():
            # Invalid move, penalize and end episode
            reward = -5 # Large penalty for invalid moves
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
        #print("------------------------------------------------------------------------------------")
        # Calculate strategic rewards
        strategic_reward += self._calculate_bridge_reward(coordinates, original_player)
        #print("Reward after bridge ", strategic_reward)
        strategic_reward += self._calculate_center_control(coordinates)
        #print("Reward after center control ", strategic_reward)
        strategic_reward += self._calculate_chain_reward(coordinates, original_player)
        #print("Reward after chain reward ", strategic_reward)

        self.hex_game.evaluate() # Check for game end (updates self.hex_game.winner)


        if self.hex_game.winner != 0:
            terminated = True
            #print("Winner is - check before opponent: ", self.hex_game.winner)
            #print("Original player is ", original_player)
            final_reward = self.get_final_reward(original_player)

        # if there is an opponent policy defined
        elif not terminated and self.opponent_policy is not None and self.hex_game.player == -original_player:
            # Opponent makes a move
            board = self.hex_game.board
            valid_actions = self.hex_game.get_action_space()

            opponent_action_coords = self.opponent_policy(board, valid_actions)
            self.hex_game.move(opponent_action_coords)
            self.hex_game.evaluate()

            # check if someone won
            if self.hex_game.winner != 0:
                terminated = True
                #print("Winner is: ", self.hex_game.winner)
                #print("Original player is ", original_player)
                final_reward = self.get_final_reward(original_player)


        reward = final_reward + strategic_reward
        #print("Reward final + strategic ", reward)
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