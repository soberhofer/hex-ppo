import random
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hex_engine import hexPosition

WITH_REWARD_SHAPING = False # set to false if no reward shaping wanted set in submission agent
#WITH_MOVE_PENALTY = True
WITH_MOVE_PENALTY = True # set in submission agent
if WITH_REWARD_SHAPING:
    MAX_REWARD = 2.0 # better difference between winning with reward shaping and having good moves, but still losing
else:
    MAX_REWARD = 1.0

if WITH_MOVE_PENALTY:
    MOVE_PENALTY_SINGLE_VALUE = 0.002  # should be in good relation to max reward (i.e., if max reward 1, 0.002 is better)
else:
    MOVE_PENALTY_SINGLE_VALUE = 0.0

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
        '''
        Determine bridge reward w. r. t. strategic advantage and bridge validity
        '''
        reward = 0
        player_stones = self._get_player_stones(player)

        for stone in player_stones:
            if stone == coordinates:
                continue

            if self._is_valid_hex_bridge(coordinates, stone):
                bridge_value = self._evaluate_bridge_strategic_value(coordinates, stone, player)
                reward += bridge_value
        return min(reward, 1.0)

    def _evaluate_bridge_strategic_value(self, pos1, pos2, player):
        """
        evaluate whether bridge counts for actual progress to goal side, connects critical areas or blocks an opponent
        """
        base_value = self.bridge_reward_value

        position_weight = self._get_target_side_weight(pos1, pos2, player)
        progress_bonus = self._calculate_connection_progress(pos1, pos2, player)
        defensive_bonus = self._calculate_defensive_value(pos1, pos2, player)
        total_value = base_value * position_weight + progress_bonus + defensive_bonus

        return total_value

    def _get_target_side_weight(self, pos1, pos2, player):
        """
        weight bridge based on how close to goal side of player
        player 1 top to bottom, player two left to right
        """
        size = self.hex_game.size
        base_weight = 1.0

        if player == 1:  # top to bottom
            # measure distance to goal sides
            top_distance1 = pos1[0]  # distance to top
            top_distance2 = pos2[0]
            bottom_distance1 = (size - 1) - pos1[0]  # distance to bottom
            bottom_distance2 = (size - 1) - pos2[0]

            # Bonus für Nähe zu oberer Seite
            closest_to_top = min(top_distance1, top_distance2)
            if closest_to_top <= 1:
                base_weight += 0.4  # Sehr nah an oberer Seite
            elif closest_to_top <= 2:
                base_weight += 0.2  # Mäßig nah an oberer Seite

            # Bonus für Nähe zu unterer Seite
            closest_to_bottom = min(bottom_distance1, bottom_distance2)
            if closest_to_bottom <= 1:
                base_weight += 0.4  # Sehr nah an unterer Seite
            elif closest_to_bottom <= 2:
                base_weight += 0.2  # Mäßig nah an unterer Seite

            # Span-Bonus: Belohnt Bridges die verschiedene Reihen verbinden
            row_span = abs(pos1[0] - pos2[0])
            span_bonus = min(row_span / (size * 0.5), 0.3)  # Max 0.3 Bonus

            return base_weight + span_bonus

        else:  # Horizontal verbinden (links-rechts)
            # Nähe zu den Zielseiten messen
            left_distance1 = pos1[1]  # Abstand zu linker Seite (col 0)
            left_distance2 = pos2[1]
            right_distance1 = (size - 1) - pos1[1]  # Abstand zu rechter Seite
            right_distance2 = (size - 1) - pos2[1]

            # Bonus für Nähe zu linker Seite
            closest_to_left = min(left_distance1, left_distance2)
            if closest_to_left <= 1:
                base_weight += 0.4
            elif closest_to_left <= 2:
                base_weight += 0.2

            # Bonus für Nähe zu rechter Seite
            closest_to_right = min(right_distance1, right_distance2)
            if closest_to_right <= 1:
                base_weight += 0.4
            elif closest_to_right <= 2:
                base_weight += 0.2

            # Span-Bonus: Belohnt Bridges die verschiedene Spalten verbinden
            col_span = abs(pos1[1] - pos2[1])
            span_bonus = min(col_span / (size * 0.5), 0.3)  # Max 0.3 Bonus

            return base_weight + span_bonus

    def _calculate_connection_progress(self, pos1, pos2, player):
        """
        reward bridges that actually connect to goal side
        """
        size = self.hex_game.size

        if player == 1:  # top to bottom
            # measure position to goal side
            top_distance = min(pos1[0], pos2[0])  # top distance
            bottom_distance = min(size - 1 - pos1[0], size - 1 - pos2[0])  # bottom distance

            # bonus if it brings together sides
            if top_distance <= 2 or bottom_distance <= 2:
                return 0.1  # Nähe zu Zielseite
            elif top_distance <= 3 and bottom_distance <= 3:
                return 0.05  # Moderate Nähe zu beiden Seiten

        else:  # Horizontal
            left_distance = min(pos1[1], pos2[1])
            right_distance = min(size - 1 - pos1[1], size - 1 - pos2[1])

            if left_distance <= 2 or right_distance <= 2:
                return 0.1
            elif left_distance <= 3 and right_distance <= 3:
                return 0.05

        return 0

    def _calculate_defensive_value(self, pos1, pos2, player):
        """
        check if bridge blocks opponent plan
        """
        opponent = 3 - player  # 1->2, 2->1
        size = self.hex_game.size

        # check if bridge blocks opponents
        blocking_value = 0

        # Bewerte beide Bridge-Positionen
        for pos in [pos1, pos2]:
            row, col = pos

            if opponent == 1:  # (top-bottom)
                # block critical central ways
                center_row = size // 2
                distance_from_center_row = abs(row - center_row) / (size // 2)

                # bonus if close to center - blocks vertical way
                if distance_from_center_row < 0.4:
                    blocking_value += 0.15
                elif distance_from_center_row < 0.7:
                    blocking_value += 0.1

            else:  # (left-right)
                center_col = size // 2
                distance_from_center_col = abs(col - center_col) / (size // 2)

                # bonus if close to center - blocks vertical way
                if distance_from_center_col < 0.4:
                    blocking_value += 0.15
                elif distance_from_center_col < 0.7:
                    blocking_value += 0.1

        # add bonus for central bridge that blocks opponent
        avg_pos = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
        center = size // 2
        distance_from_center = max(
            abs(avg_pos[0] - center),
            abs(avg_pos[1] - center)
        ) / center

        if distance_from_center < 0.6:
            blocking_value += 0.05

        return min(blocking_value, 0.4)


    def _is_valid_hex_bridge(self, pos1, pos2):
        """
        check if two positions can be hex bridge --> 2 stones with exactly 2 empty neighbours
        """
        neighbors1 = set(self.hex_game._get_adjacent(pos1))
        neighbors2 = set(self.hex_game._get_adjacent(pos2))

        common_neighbors = neighbors1 & neighbors2

        empty_common = [
            coord for coord in common_neighbors
            if self.hex_game.board[coord[0]][coord[1]] == 0
        ]

        return len(empty_common) == 2

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

        return 0.1 * math.log(1 + chain_length)



    def get_final_reward(self, original_player):
        if WITH_REWARD_SHAPING:
            base = MAX_REWARD if self.hex_game.winner == original_player else -MAX_REWARD
        else:
            base = MAX_REWARD if self.hex_game.winner == original_player else -MAX_REWARD
        #print("BASE REWARD: ", base)
        move_penalty = -MOVE_PENALTY_SINGLE_VALUE * self.hex_game.move_count # idea: faster games get better final rewards
        #print("MOVE PENALTY: ", move_penalty)
        return base + move_penalty

    def _calculate_strategic_rewards(self, coordinates, player):
        """Weighted rewards"""
        rewards = {
            'bridge': self._calculate_bridge_reward(coordinates, player) * 0.3,
            'chain': self._calculate_chain_reward(coordinates, player) * 0.2,
            'connectivity': self._reward_connection_progress(coordinates, player) * 0.4,
        }
        return sum(rewards.values())

    def _reward_connection_progress(self, coords, player):
        """
        Reward actual progress toward victory
        """
        connected_group = self._get_connected_group(coords, player)

        if not connected_group:
            return 0.0

        # calculate progress based on connected groupe
        progress_score = self._calculate_group_progress(connected_group, player)

        # bonus if near critical area
        critical_bonus = self._calculate_critical_area_bonus(coords, connected_group, player)

        total_reward = progress_score + critical_bonus

        return min(total_reward, 1.0)

    def _get_connected_group(self, start_coords, player):
        """
        get all stones connected with start_coords
        """
        visited = set()
        queue = [start_coords]
        connected_group = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            connected_group.append(current)

            # check all neighbors
            for neighbor in self.hex_game._get_adjacent(current):
                if (neighbor not in visited and
                        self.hex_game.board[neighbor[0]][neighbor[1]] == player):
                    queue.append(neighbor)

        return connected_group

    def _calculate_group_progress(self, connected_group, player):
        """
        calculate how close the connected group is to a victory
        """
        if not connected_group:
            return 0.0

        size = self.hex_game.size

        if player == 1:  # top to bottom
            # find min and max row of group
            rows = [pos[0] for pos in connected_group]
            min_row = min(rows)
            max_row = max(rows)

            # how much of the board does this group span
            span_progress = (max_row - min_row) / (size - 1)

            # bonus if close to top or bottom
            top_proximity = max(0, (2 - min_row) / 2)
            bottom_proximity = max(0, (2 - (size - 1 - max_row)) / 2)

            progress = 0.4 * span_progress + 0.2 * top_proximity + 0.2 * bottom_proximity

            # extra bonus if groups touch both sides or are close to
            if min_row == 0 and max_row == size - 1:
                progress += 0.5  # victory!
            elif min_row <= 1 and max_row >= size - 2:
                progress += 0.3  # very close to victory

        else:  # left to right
            # get min and max col of connected group
            cols = [pos[1] for pos in connected_group]
            min_col = min(cols)
            max_col = max(cols)

            # how much of the board does this group span
            span_progress = (max_col - min_col) / (size - 1)

            # bonus if close to left/right
            left_proximity = max(0, (2 - min_col) / 2)
            right_proximity = max(0, (2 - (size - 1 - max_col)) / 2)

            progress = 0.4 * span_progress + 0.2 * left_proximity + 0.2 * right_proximity

            #  bonus for groups that touch both sides or are very close
            if min_col == 0 and max_col == size - 1:
                progress += 0.5  # victory!
            elif min_col <= 1 and max_col >= size - 2:
                progress += 0.3  # very close to

        return progress

    def _calculate_critical_area_bonus(self, coords, connected_group, player):
        """
        bonus for stones in critical areas of the board
        """
        size = self.hex_game.size
        bonus = 0.0

        if player == 1:  # top to bottom
            row = coords[0]

            # bonus if stones close gaps in vertical direction
            group_rows = set(pos[0] for pos in connected_group)

            # check if the stone closes gap or blocks opponent
            min_group_row = min(group_rows)
            max_group_row = max(group_rows)

            if row < min_group_row or row > max_group_row:
                # creates bigger group in goal direction
                if row < min_group_row and min_group_row > 2:  # top
                    bonus += 0.1
                elif row > max_group_row and max_group_row < size - 3:  #bottom
                    bonus += 0.1

            # bonus for stones in critical areas (middle rows)
            middle_zone_distance = abs(row - size // 2) / (size // 2)
            if middle_zone_distance < 0.4:  # middle 40 % section
                bonus += 0.05

        else:  # left to right
            col = coords[1]

            group_cols = set(pos[1] for pos in connected_group)

            min_group_col = min(group_cols)
            max_group_col = max(group_cols)

            if col < min_group_col or col > max_group_col:
                if col < min_group_col and min_group_col > 2:  # left
                    bonus += 0.1
                elif col > max_group_col and max_group_col < size - 3:  # right
                    bonus += 0.1

            middle_zone_distance = abs(col - size // 2) / (size // 2)
            if middle_zone_distance < 0.4:
                bonus += 0.05

        return bonus

    def _calculate_opponent_chain_length(self, start_coord, opponent):
        '''
        get the opponents chain length to evaluate strategic advantage of own move
        '''
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
        '''
        add reward if this could block opponent connection
        '''
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
        '''
        add reward if this chain blocks an opponent
        '''
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


    def step(self, action, current_payer_in_game, ppo_agent_player_id):
        coordinates = self.hex_game.scalar_to_coordinates(action)
        original_player = current_payer_in_game

        # Check if the move is valid
        if coordinates not in self.hex_game.get_action_space():
            # Invalid move, penalize and end episode
            reward = -1 # Large penalty for invalid moves
            terminated = True
            truncated = False # Not used in this env
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Make the move
        self.hex_game.move(coordinates) # board is updated here, self.hex_game.player is flipped

        # initialize rewards
        strategic_reward = 0.0
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
            #print("Winner is - check before opponent: ", self.hex_game.winner)
            #print("Original player is ", original_player)
            final_reward = self.get_final_reward(original_player)
            #print("FINAL REWARD IS: ", final_reward)


        # if there is an opponent policy defined and its the opponents turn
        elif not terminated and self.opponent_policy is not None and self.hex_game.player != ppo_agent_player_id:
            #print(f"OPPONENT IS: {self.opponent_name},                  HEX GAME PLAYER AFTER MOVE: {self.hex_game.player} current player per parameter {current_payer_in_game}")
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
                #print("FINAL REWARD IS: ", final_reward)

        # strategic rewards should be less important at the end of the game test
        #progress = self.hex_game.move_count / (self.size**2)
        #strategic_weight = 0.6 * (1 - progress)
        #little_random_noise_for_exploration = 0.1*random.uniform(-0.1, 0.1)
        reward = final_reward + strategic_reward


        #reward = final_reward + strategic_reward
        #print("Reward final + strategic ", reward)
        observation = self._get_obs()
        info = self._get_info()

        """"
        print(    f"[STEP] strategic reward: {strategic_reward} , final_reward: {final_reward}")#, * {strategic_weight} noise: {little_random_noise_for_exploration}")
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


    def set_opponent_policy(self, policy_fn, name):
        self.opponent_name = name
        self.opponent_policy = policy_fn