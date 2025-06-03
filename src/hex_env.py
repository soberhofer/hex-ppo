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
            truncated = False # Not used in this env
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        original_player = self.hex_game.player
        
        # Make the move
        self.hex_game.move(coordinates) # board is updated here, self.hex_game.player is flipped

        intermediate_reward = 0.0
        
        # --- Bridge Detection ---
        # The stone just placed is at 'coordinates' by 'original_player'.
        # Check if this move formed any new bridges.
        newly_placed_stone = coordinates
        
        # Iterate over all stones of the player who just moved (original_player)
        # to find potential existing stones (S1) that could form a bridge with newly_placed_stone (S2)
        for r_idx in range(self.hex_game.size):
            for c_idx in range(self.hex_game.size):
                if self.hex_game.board[r_idx][c_idx] == original_player:
                    existing_stone = (r_idx, c_idx)
                    if existing_stone == newly_placed_stone:
                        continue # Don't check a stone against itself

                    # Find common neighbors of newly_placed_stone and existing_stone
                    neighbors_new = set(self.hex_game._get_adjacent(newly_placed_stone))
                    neighbors_existing = set(self.hex_game._get_adjacent(existing_stone))
                    
                    common_neighbors_coords = list(neighbors_new.intersection(neighbors_existing))
                    
                    empty_common_neighbors = []
                    if len(common_neighbors_coords) >= 2: # Need at least two common neighbors for a bridge
                        for cn_coord in common_neighbors_coords:
                            if self.hex_game.board[cn_coord[0]][cn_coord[1]] == 0: # Check if common neighbor is empty
                                empty_common_neighbors.append(cn_coord)
                    
                    # If S_new and S_old share exactly two common empty neighbors, a bridge is formed.
                    if len(empty_common_neighbors) == 2:
                        intermediate_reward += self.bridge_reward_value
                        # Note: A single move could complete multiple bridges.
                        # This will sum rewards if multiple bridges are formed.
                        # To reward only once per step if *any* bridge is formed, break here or set a flag.
                        # For now, summing is fine.
        # --- End Bridge Detection ---

        self.hex_game.evaluate() # Check for game end (updates self.hex_game.winner)

        final_reward = 0.0
        terminated = False
        truncated = False # Not used in this env

        if self.hex_game.winner != 0:
            terminated = True
            if self.hex_game.winner == original_player:
                final_reward = 5.0  # Player who made the move won
            else:
                final_reward = -5.0 # Player who made the move lost (opponent won)
        
        reward = final_reward + intermediate_reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.hex_game.print()

    def close(self):
        pass
