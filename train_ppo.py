import torch
from src.hex_env import HexEnv
from src.ppo_agent import PPOAgent, RolloutMemory
from src.ppo_model import ActorCritic # For evaluation
from hex_engine import hexPosition # For evaluation
from torch.optim.lr_scheduler import StepLR # Import StepLR
import numpy as np
import os
import random # For random agent in evaluation and mixed training
import time
import copy

# Hyperparameters
TEMPERATURE = 1.4
FINAL_TEMPERATURE = 1.0
HEX_BOARD_SIZE = 7
INITIAL_LEARNING_RATE = 0.001 # this is the learning rate up until linear warm up goes
GAMMA = 0.99
K_EPOCHS = 10
EPS_CLIP = 0.2
GAE_LAMBDA = 0.95           # bias in advantage estimates
ENTROPY_COEF_INITIAL = 0.03 # higher means more exploration in the beginning, gets reduced throughout training with each update in ppo agent
ENTROPY_COEF_FINAL = 0.001

MAX_TOTAL_TIMESTEPS = 5000000  # Total timesteps to train for
TIMESTEPS_PER_BATCH = 2048   # Timesteps to collect per batch before updating
UPDATES_PER_EVAL = 10        # Evaluate model every X updates (e.g., 50 updates * 2048 steps/update = ~100k steps)
UPDATES_PER_SAVE = 250       # Save model every X updates (e.g., 250 updates * 2048 steps/update = ~500k steps)
# LR Scheduler: step_size is now in terms of number of updates
LR_SCHEDULER_STEP_SIZE = 50 # Decay LR every X updates (e.g. 50 updates)
LR_SCHEDULER_GAMMA = 0.9    # Multiplicative factor of LR decay
WARMUP_EPOCHS = int(0.05 * MAX_TOTAL_TIMESTEPS) # 5% of overall total steps

RANDOM_OPPONENT_RATIO = 0.3 # Play against random opponent for this fraction of episodes in the beginning

NUM_EVAL_GAMES = 100 # Number of games for periodic evaluation
MODEL_DIR = "./models"

frozen_agent_list = []

class Opponents:
    RANDOM = "random"
    SELF = "self"
    FROZEN_SELF= "frozen_self"
    GREEDY="greedy"


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # auto-tuner for CNNs
    torch.set_float32_matmul_precision('high')

# --- Random Agent for Evaluation & Mixed Training ---
def random_opponent_action_logic(game_engine_instance):
    action_set_tuples = game_engine_instance.get_action_space()
    if not action_set_tuples: # Should not happen in a valid game state before end
        return None # Or handle error appropriately
    chosen_coords = random.choice(action_set_tuples)
    return game_engine_instance.coordinate_to_scalar(chosen_coords)


def greedy_action(board: np.ndarray, valid_actions: list, policy_net: torch.nn.Module, device: torch.device, env: HexEnv):
    """
    Selects the action with highest predicted probability from the policy network
    Args:
        board: Current Hex board state (2D numpy array)
        valid_actions: List of valid (row,col) coordinates
        policy_net (torch.nn.Module): The PPO actor network.
        device (torch.device): CPU/GPU device.
        env (HexEnv): Environment instance (needed for coordinate conversions).

    Returns:
        (row, col) coordinates of the greedy action
    """
    # can be array or torch.Tensor, but MUST be tensor for policy network
    if type(board) != torch.Tensor:
        board = torch.FloatTensor(board).unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        # get logits from policy network (model.forward())
        action_logits, _ = policy_net(board)

        valid_action_indices = []
        for a in valid_actions:
            if isinstance(a, int):
                valid_action_indices.append(a)
            else:
                valid_action_indices.append(env.hex_game.coordinate_to_scalar(a))

        # exclude invalid actions to get excluded by softmax
        mask = torch.full(action_logits.shape, -float('inf'), device=device)
        if valid_action_indices:  # Ensure there are valid actions
            mask[0, valid_action_indices] = 0

        # apply the mask & get action probalities
        masked_logits = action_logits + mask

        probs = torch.nn.functional.softmax(masked_logits, dim=-1)
        best_action_index = torch.argmax(probs).item()

    return best_action_index

def ppo_action_from_policy(board, valid_actions: list, policy_net: torch.nn.Module, device: torch.device, env: HexEnv):
    """
     Select an action using a PPO policy network, constrained to valid actions.

     Args:
         board: Current game board - can be array or torch tensor
         valid_actions (list): List of valid actions as (row, col) tuples.
         policy_net (torch.nn.Module): The PPO actor network.
         device (torch.device): CPU/GPU device.
         env (HexEnv): Environment instance (needed for coordinate conversions).

     Returns:
         Tuple (row, col): The selected action in coordinates.
     """
    # can be array or torch.Tensor, but MUST be tensor for policy network
    if type(board) != torch.Tensor:
        board = torch.FloatTensor(board).unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        # get logits from policy network (model.forward())
        action_logits, _ = policy_net(board)

        valid_action_indices = []
        for a in valid_actions:
            if isinstance(a, int):
                valid_action_indices.append(a)
            else:
                valid_action_indices.append(env.hex_game.coordinate_to_scalar(a))

        # exclude invalid actions to get excluded by softmax
        mask = torch.full(action_logits.shape, -float('inf'), device=device)
        if valid_action_indices: # Ensure there are valid actions
            mask[0, valid_action_indices] = 0

        # apply the mask & get action probalities
        masked_logits = action_logits + mask
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)

        # sample an action from the masked distribution
        dist = torch.distributions.Categorical(probs)
        action_scalar = dist.sample().item()
        # Check for NaN in probs, can happen if all logits are -inf (no valid moves, though env should prevent this)
        if torch.isnan(probs).any():
            # Fallback to random action if probs are NaN (should ideally not happen if valid_actions is managed well)
            print("Warning: NaN in probabilities during evaluation, choosing random action.")
            chosen_action_tuple = random.choice(valid_actions) if valid_actions else env.hex_game.scalar_to_coordinates(0) # Fallback
            action_coords = chosen_action_tuple
        else:
            dist = torch.distributions.Categorical(probs)
            action_scalar = dist.sample().item()
            action_coords = env.hex_game.scalar_to_coordinates(action_scalar)
        return action_coords

best_results_for_each_agent = {}

def update_best_agent_stats(random_win_rate: float, timesteps_collected: int, current_agent: str, agent_key: str) -> None:
    if agent_key not in best_results_for_each_agent:
        best_results_for_each_agent[agent_key] = {
            "win_rate": 0.0,
            "agent": None,
            "timesteps": 0,
            "updates": 0
        }

    if random_win_rate > best_results_for_each_agent[agent_key]["win_rate"]:
        best_results_for_each_agent[agent_key]["win_rate"] = random_win_rate
        best_results_for_each_agent[agent_key]["agent"] = current_agent
        best_results_for_each_agent[agent_key]["timesteps"] = timesteps_collected
        best_results_for_each_agent[agent_key]["updates"] += 1


def evaluate_mixed(ppo_policy_net, device, env: HexEnv, time_steps_collected: int, num_games=100):
    """
    Evaluate against 50 random games and 50 x agents in frozen_agents games against frozen agents
    """
    assert len(frozen_agent_list) > 0, "At least one frozen agent .pth file is required for evaluation."

    print(f"\n--- Evaluating PPO Agent vs Random (50) + Frozen Agents ({len(frozen_agent_list)} total, 50 games) ---")

    stats = {
        Opponents.RANDOM: {"wins": 0, "games": 0},
        Opponents.FROZEN_SELF: [{"wins": 0, "games": 0} for _ in frozen_agent_list]
    }

    ppo_policy_net.eval()
    game_engine = hexPosition(size=HEX_BOARD_SIZE)

    total_games = 50 + 50 * len(frozen_agent_list)

    for i in range(total_games):
        game_engine.reset()
        ppo_as_player = 1 if i % 2 == 0 else -1
        game_engine.player = 1

        if i < 50:
            opponent_type = Opponents.RANDOM
            frozen_idx = None
        else:
            # From 50 onward: evaluate each frozen agent 50 times
            opponent_idx = (i - 50) // 50  # 0, 1, 2, ...
            opponent_type = f"frozen_{opponent_idx}"

        while game_engine.winner == 0:
            current_board = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            valid_actions = game_engine.get_action_space()

            is_ppo_turn = (game_engine.player == 1 and ppo_as_player == 1) or \
                          (game_engine.player == -1 and ppo_as_player == -1)

            if is_ppo_turn:
                action = ppo_action_from_policy(current_board, valid_actions, ppo_policy_net, device, env)
            else:
                if opponent_type == Opponents.RANDOM:
                    action = random.choice(valid_actions)
                else:
                    frozen_agent = frozen_agent_list[opponent_idx][1]
                    action = ppo_action_from_policy(current_board, valid_actions, frozen_agent.policy, device, env)

            game_engine.move(action)
            game_engine.evaluate()

        # Count result
        if game_engine.winner == ppo_as_player:
            if opponent_type == Opponents.RANDOM:
                stats[Opponents.RANDOM]["wins"] += 1
            else:
                idx = int(opponent_type.split("_")[1])
                stats[Opponents.FROZEN_SELF][idx]["wins"] += 1

        if opponent_type == Opponents.RANDOM:
            stats[Opponents.RANDOM]["games"] += 1
        else:
            idx = int(opponent_type.split("_")[1])
            stats[Opponents.FROZEN_SELF][idx]["games"] += 1

    # --- Report results ---
    print("\n--- Evaluation Results ---")

    random_wins, random_games = stats[Opponents.RANDOM]["wins"], stats[Opponents.RANDOM]["games"]
    random_win_rate = random_wins / max(1, random_games)
    print(f"Random Opponent: {random_wins} / {random_games} wins "
          f"({random_win_rate:.2f})")

    update_best_agent_stats(random_win_rate, time_steps_collected, Opponents.RANDOM, Opponents.RANDOM)

    agents = [agent[0] for agent in frozen_agent_list]
    for i, frozen_stat in enumerate(stats[Opponents.FROZEN_SELF]):

        wins, games = frozen_stat["wins"], frozen_stat["games"]
        win_rate = wins / max(1, games)
        agent = str(agents[i])

        print(f"Frozen Agent {agent}: "
              f"{wins} / {games} wins ({win_rate:.2f})")

        update_best_agent_stats(win_rate, time_steps_collected, Opponents.FROZEN_SELF+agent, agent)

    print("--- Evaluation Finished ---\n")
    ppo_policy_net.train()
    return stats





# --- Evaluation Function (integrated) ---
def evaluate_against_random(ppo_policy_net, device,  env: HexEnv, num_games=NUM_EVAL_GAMES,):
    print(f"\n--- Evaluating PPO Agent vs Random Agent for {num_games} games ---")
    ppo_wins = 0
    game_engine = hexPosition(size=HEX_BOARD_SIZE)
    ppo_policy_net.eval() # Ensure ppo_policy_net is in eval mode

    for i in range(num_games):
        game_engine.reset()
        if i % 2 == 0:
            current_player1_is_ppo = True
            ppo_plays_as_player = 1 # PPO is player 1 (White)
        else:
            current_player1_is_ppo = False
            ppo_plays_as_player = -1 # PPO is player -1 (Black)

        while game_engine.winner == 0:

            # convert board to tensor, add batch & channel dimensions
            current_board_for_nn = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            action_coords = None

            is_ppo_turn_now = (game_engine.player == 1 and current_player1_is_ppo) or \
                              (game_engine.player == -1 and not current_player1_is_ppo)

            if is_ppo_turn_now:
                with torch.no_grad():
                    valid_actions_tuples = game_engine.get_action_space()
                    action_coords = ppo_action_from_policy(current_board_for_nn, valid_actions_tuples, ppo_policy_net, device, env)

            else: # Random agent's turn
                action_coords = random.choice(game_engine.get_action_space()) # random_agent_eval simplified
                # action_coords = random_agent_eval(game_engine.board, game_engine.get_action_space())
            
            if action_coords is None: # Should not happen if logic is correct
                print("Error: action_coords is None. Defaulting to random.")
                valid_actions = game_engine.get_action_space()
                action_coords = random.choice(valid_actions) if valid_actions else (0, 0)

            game_engine.move(action_coords)
            game_engine.evaluate()

        if game_engine.winner == ppo_plays_as_player:
            ppo_wins += 1
            
    win_rate = (ppo_wins / num_games) * 100
    print(f"PPO Agent win rate vs Random: {win_rate:.2f}% ({ppo_wins}/{num_games})")
    print("--- Evaluation Finished ---")
    ppo_policy_net.train() # Set policy back to train mode
    return win_rate

def freeze_agent_and_reset_policy(frozen_agent, agent, env, device, num_updates):
    """"
        Freezes the agent and resets the policy, so that ppo agent can play against older versions of itself.
    """
    frozen_agent = copy.deepcopy(agent)

    # TODO: check if this can lead to memory problems
    if len(frozen_agent_list) == 6:
        frozen_agent_list.pop(0)    # remove first, currently want to keep three agents only

    frozen_agent_list.append((num_updates, frozen_agent))

    env.set_opponent_policy(lambda b, va: ppo_action_from_policy(b, va, frozen_agent.policy, device, env))
    print(f"Opponent replaced with frozen snapshot at update {num_updates}")

def ppo_turn(agent, state, valid_actions, temperature):
    action_scalar_ppo, log_prob_ppo, _ = agent.select_action(state, valid_actions, temperature)
    return action_scalar_ppo, log_prob_ppo

def determine_opponent(with_random: bool, with_periodic_self: bool, rand_val: float, random_opponent_ratio: float, frozen_self_ratio: float,  greedy_ratio: float, self_ratio: float, force_first = False):
    if force_first:
        return Opponents.FROZEN_SELF
    #print(f"OPPONENT RANDOM: {rand_val} / {adjusted_random_ratio}")
    #print(f"OPPONENT FROZEN SELF: {rand_val} / {adjusted_random_ratio + frozen_self_ratio}")
    #print(f"OPPONENT GREEDY: {rand_val} / {adjusted_random_ratio + frozen_self_ratio + greedy_ratio}")
    #print(f"OPPONENT SELF: {rand_val} / {adjusted_random_ratio + frozen_self_ratio + greedy_ratio}")

    random_cutoff = random_opponent_ratio
    frozen_cutoff = random_cutoff + frozen_self_ratio
    greedy_cutoff = frozen_cutoff + greedy_ratio

    if with_random and rand_val < random_cutoff:
        return Opponents.RANDOM
    elif with_periodic_self and rand_val < frozen_cutoff:
        return Opponents.FROZEN_SELF
    elif rand_val < greedy_cutoff:
        return Opponents.GREEDY
    else:
        return Opponents.SELF


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU device for training.")
    #device = torch.device("cpu")  # FORCE CPU
    print(f"Training will run on: {device}")
    return device


def get_scheduler(agent):
    # lr_scheduler = StepLR(agent.optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)

    # Define both schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(agent.optimizer,
                                                         start_factor=0.01,
                                                         total_iters=WARMUP_EPOCHS
                                                         # iterations until which the initial LR is reached
                                                         )

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        patience=10,  # iterations the scheduler waits until it reduces the LR
        factor=0.9,  # factor the LR gets multiplicated with
        min_lr=1e-6,  # min LR that will be kept as lower boundary
        threshold=1e-4,
    )
    return warmup_scheduler, plateau_scheduler

def update_scheduler(num_updates, warmup_scheduler, plateau_scheduler, v_loss, avg_reward):
    if num_updates < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        # plateau scheduler needs step
        plateau_scheduler.step(avg_reward)
    # lr_scheduler.step()

def get_temperature(total_timesteps_collected):
    """
        Measurement to support more entropy, i.e., more exploration --> agent as is gets stuck too fast
    """
    progress = total_timesteps_collected / MAX_TOTAL_TIMESTEPS
    return FINAL_TEMPERATURE + (TEMPERATURE - FINAL_TEMPERATURE) * np.exp(-5 * progress)


def get_random_opp_prob(current_step, initial_prob=RANDOM_OPPONENT_RATIO):
    # start at RANDOM_OPPONENT_RATIO, decrease over time
    return initial_prob * (1 - current_step / MAX_TOTAL_TIMESTEPS)


def get_entropy_coef(current_step, initial=ENTROPY_COEF_INITIAL, final=ENTROPY_COEF_FINAL):
    # entropy coeff annealing --> supports more exploration in the beginning, reduced throughout training
    return initial - (initial - final) * (current_step / MAX_TOTAL_TIMESTEPS)


def get_ratios(total_timesteps_collected: int):
    # Determine opponent for the new episode
    progress = min(total_timesteps_collected / MAX_TOTAL_TIMESTEPS, 1.0)

    if progress < 1/3:
        random_ratio = RANDOM_OPPONENT_RATIO
        greedy_ratio = 0.0
    elif progress < 2/3:
        random_ratio = RANDOM_OPPONENT_RATIO / 2
        greedy_ratio = RANDOM_OPPONENT_RATIO / 2
    else:
        random_ratio = 0.0
        greedy_ratio = RANDOM_OPPONENT_RATIO

    remaining = max(0.0, 1.0 - random_ratio - greedy_ratio)

    self_ratio = 0.3 * remaining
    frozen_ratio = 0.7 * remaining

    rand_val = random.random()
    return rand_val, random_ratio, greedy_ratio, frozen_ratio, self_ratio

def train(with_periodic_self: bool = True, with_random: bool = True):
    device = get_device()
    env = HexEnv(size=HEX_BOARD_SIZE)
    obs_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    agent = PPOAgent(obs_shape, action_space_size, INITIAL_LEARNING_RATE, GAMMA, K_EPOCHS, EPS_CLIP, GAE_LAMBDA, device,
                     torch.cuda.is_available())

    memory = RolloutMemory(device)

    if with_periodic_self:
        frozen_agent = copy.deepcopy(agent)
        # try self play instead of HEXGAME agent
        def self_play_opponent(board, valid_actions):
            return ppo_action_from_policy(board, valid_actions, agent.policy, device, env)

        env.set_opponent_policy(self_play_opponent)
        freeze_agent_and_reset_policy(frozen_agent, agent, env, device, action_space_size)

    #lr_scheduler = StepLR(agent.optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    warmup_scheduler, plateau_scheduler = get_scheduler(agent)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Starting PPO training for Hex for {MAX_TOTAL_TIMESTEPS} timesteps...")
    print("Training on #experiments Branch")
    print(f"Batch size: {TIMESTEPS_PER_BATCH}, Updates per batch: {K_EPOCHS}")

    if with_random:
        print(f"Playing against random opponent with initially {RANDOM_OPPONENT_RATIO*100}% probability.")

    total_timesteps_collected = 0
    num_updates = 0
    all_episode_rewards = []
    state, info = env.reset()
    current_episode_reward_accumulator = 0.0

    rand_val, random_ratio, greedy_ratio, frozen_ratio, self_ratio = get_ratios(total_timesteps_collected)
    opponent_type = determine_opponent(with_random, with_periodic_self, rand_val, random_ratio, frozen_ratio, greedy_ratio, self_ratio)

    ppo_agent_player_id = 1 # Default, will be set if opponent is random
    if opponent_type in [Opponents.RANDOM, Opponents.FROZEN_SELF, Opponents.GREEDY]:
        ppo_agent_player_id = random.choice([1, -1])

    while total_timesteps_collected < MAX_TOTAL_TIMESTEPS:

        # ---- Do update loop
        for _ in range(TIMESTEPS_PER_BATCH):
            valid_actions = info["valid_actions"]
            action_scalar_to_env = -1 # Placeholder
            
            is_ppo_turn_for_memory = False
            current_player_in_game = env.hex_game.player # Player whose turn it is in hex_engine
            #print(f"[BEFORE STEP] Current player: {env.hex_game.player}, action: {action_scalar_to_env}")

            # ---- player or opponent make a move
            if current_player_in_game == ppo_agent_player_id or opponent_type == Opponents.SELF:
                #if current_player_in_game == 1:
                    #print("CURRENT PLAYER IS SELF")
                #if opponent_type == "self":
                    #print("OPPONENT IS SELF")
                is_ppo_turn_for_memory = True
                temperature = get_temperature(total_timesteps_collected)
                action_scalar_to_env, log_prob_ppo = ppo_turn(agent, state, valid_actions, temperature)

            elif with_periodic_self and  opponent_type == Opponents.FROZEN_SELF:
                #print("FROZEN SELF OPPONENT")
                # train against latest saved
                action_scalar_to_env = env.hex_game.coordinate_to_scalar(ppo_action_from_policy(state, valid_actions, frozen_agent.policy, device, env)) # ignore warning, frozen agent gets initialized, if periodic self is initialized

            elif with_random and opponent_type == Opponents.RANDOM:
                #print("RANDOM OPPONENT ")
                action_scalar_to_env = random_opponent_action_logic(env.hex_game)

            elif opponent_type == Opponents.GREEDY:
                # print("GREEDY OPPONENT ")
                action_scalar_to_env = greedy_action(state, valid_actions, agent.policy, device, env) # returns scalar

            else:
                print("ERROR: This state should never be reached.")


            # ----- environment gets updated and step saved, if move was agents move
            next_state, step_reward, done, truncated, next_info = env.step(action_scalar_to_env)
            #print(
            #    f"[STEP] Action taken: {action_scalar_to_env}, reward: {step_reward}, done: {done}, next_player: {env.hex_game.player}")

            if is_ppo_turn_for_memory: # Only add to memory if PPO made the move
                memory.add(state, action_scalar_to_env, log_prob_ppo, step_reward, done or truncated)

            current_episode_reward_accumulator += step_reward # Accumulate for episode outcome logging

            state = next_state
            info = next_info
            total_timesteps_collected += 1

            # ---- game is through, update stats, reset and determine opponent for next episode
            if done or truncated:
                all_episode_rewards.append(current_episode_reward_accumulator)
                current_episode_reward_accumulator = 0.0 
                state, info = env.reset()
                rand_val = random.random()

                # Determine opponent for the new episode
                rand_val, random_ratio, progress, greedy_ratio, frozen_ratio = get_ratios(total_timesteps_collected)
                opponent_type = determine_opponent(with_random, with_periodic_self, rand_val, random_ratio, frozen_ratio, greedy_ratio, self_ratio)

                if opponent_type in [Opponents.RANDOM, Opponents.FROZEN_SELF, Opponents.GREEDY]:
                    ppo_agent_player_id = random.choice([1, -1])

            if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
                break

            #rint(f"Done step {total_timesteps_collected}, current opponent type: {opponent_type}")

        # ---- update training
        if len(memory.states) > 0:
            entropy_coef = get_entropy_coef(total_timesteps_collected)

            p_loss, v_loss, ent, combined_loss = agent.update(memory, entropy_coef)
            memory.clear_memory()
            num_updates += 1

            # LR updates --> correct scheduler should get increased
            # update_scheduler(num_updates, warmup_scheduler, plateau_scheduler, v_loss)
            # last x rewards mean
            avg_rewards = np.mean(all_episode_rewards[-15:])

            # try to update scheduler based on increasing rewards
            update_scheduler(num_updates, warmup_scheduler, plateau_scheduler, combined_loss, -avg_rewards) # neg due to minimize of plateauscheduler


            # --- log outputs every ten updates
            if num_updates % 10 == 0:
                current_lr = agent.optimizer.param_groups[0]['lr']
                avg_ep_reward_str = ""
                if len(all_episode_rewards) > 0:
                    lookback_episodes = min(50, len(all_episode_rewards))
                    avg_recent_ep_reward = np.mean(all_episode_rewards[-lookback_episodes:])
                    avg_ep_reward_str = f", Avg Ep Reward (last ~{lookback_episodes}): {avg_recent_ep_reward:.2f}"

                print(f"Update {num_updates}, Timesteps: {total_timesteps_collected}, LR: {current_lr:.7f}{avg_ep_reward_str}")
                print(f"  Losses: Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Entropy: {ent:.4f}")

            # --- periodic evaluation
            if num_updates > 0 and num_updates % UPDATES_PER_EVAL == 0:
                evaluate_mixed(agent.policy, device, env, total_timesteps_collected )
                #evaluate_against_random(agent.policy, device, env, NUM_EVAL_GAMES)

            # --- periodic model saving
            if num_updates > 0 and num_updates % UPDATES_PER_SAVE == 0:
                model_path = os.path.join(MODEL_DIR, f"ppo_hex_agent_update_{num_updates}_steps_{total_timesteps_collected}.pth")
                torch.save(agent.policy.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            # --- periodic replacement with frozen snapshot --> updates the opponent, so that it gets smarter too
            if with_periodic_self:
                if num_updates % 200 == 0:
                    print("Replacing current frozen_self with updated version")
                    freeze_agent_and_reset_policy(frozen_agent, agent, env, device, num_updates)


        if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
            break

    env.close()
    for key, item in best_results_for_each_agent.items():
        print(f"{key} Best: ")
        for sub_key, sub_item in item.items():
            print(f"{sub_key} - {sub_item}")

    print(best_results_for_each_agent)
    print("Training finished.")

if __name__ == '__main__':
    train()
