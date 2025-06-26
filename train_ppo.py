from shutil import chown

import torch

import secrets
from src import hex_env
from src.hex_env import HexEnv
from src.ppo_agent import PPOAgent, RolloutMemory
from src.ppo_model import ActorCritic  # For evaluation
from hex_engine import hexPosition  # For evaluation
from torch.optim.lr_scheduler import StepLR  # Import StepLR
import numpy as np
import os
import random  # For random agent in evaluation and mixed training
import copy
import wandb


# different types of opponents, relevant for evaluating and curriculum learning
class Opponents:
    RANDOM = "random"
    SELF = "self"
    FROZEN_SELF = "frozen_self"
    GREEDY = "greedy"

# Hyperparameters PPO Model
TEMPERATURE = 1.0
FINAL_TEMPERATURE = 1.0
INITIAL_LEARNING_RATE = 0.0003  # this is the learning rate - STATIC NOW (apart from reducing on stagnation)
GAMMA = 0.99
K_EPOCHS = 8
EPS_CLIP = 0.15
GAE_LAMBDA = 0.92  # bias in advantage estimates
ENTROPY_COEF_INITIAL = 0.03  # higher means more exploration in the beginning, gets reduced throughout training with each update in ppo agent
ENTROPY_COEF_FINAL = 0.002
MAX_CLIPPING = 0.25
VALUE_COEF = 0.4
MAX_TOTAL_TIMESTEPS = 1500000  # Total timesteps to train for
TIMESTEPS_PER_BATCH = 2048  # Timesteps to collect per batch before updating
UPDATES_PER_EVAL = 10  # Evaluate model every X updates (e.g., 50 updates * 2048 steps/update = ~100k steps)
UPDATES_PER_SAVE = 200  # Save model every X updates (e.g., 250 updates * 2048 steps/update = ~500k steps)

# Game Parameter
HEX_BOARD_SIZE = 7

# Curriculum Learning Parameters, the final model was trained without
RANDOM_OPPONENT_RATIO_EASY = 0.3
RANDOM_OPPONENT_RATIO_MEDIUM = 0.15
RANDOM_OPPONENT_RATIO_HARD = 0.05
GREEDY_OPPONENT_RATIO_EASY = 0.1
GREEDY_OPPONENT_RATIO_MEDIUM = 0.15
GREEDY_OPPONENT_RATIO_HARD = 0.3
MAX_LEN_FROZEN_AGENTS = 3
PERIODIC_FROZEN_AGENT_UPDATE_COUNTER = 250  # every x updates, update the frozen agent list
frozen_agent_list = []

MODEL_DIR: str = "./models"
BEST_MODEL_PATH = f"{MODEL_DIR}/ppo_hex_agent_update_best_so_far.pth"

# Statistics
# (update, win_rate)
overall_best = [(0, 0.0)]

opponent_counts = {
    Opponents.RANDOM: 0,
    Opponents.SELF: 0,
    Opponents.FROZEN_SELF: 0,
    Opponents.GREEDY: 0,
}

opponent_counts_per_eval_loop = {
    Opponents.RANDOM: 0,
    Opponents.SELF: 0,
    Opponents.FROZEN_SELF: 0,
    Opponents.GREEDY: 0,
}

# GPU specific optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # auto-tuner for CNNs
    torch.set_float32_matmul_precision('high')


# --- Random Agent ---
def random_opponent_action_logic(game_engine_instance):
    """
        Selects a random action from the action space
        Args:
            game_engine_instance: Game engine instance
        Returns:
            (row, col) coordinates of the greedy action
        """

    action_set_tuples = game_engine_instance.get_action_space()
    if not action_set_tuples:  # Should not happen in a valid game state before end
        return None
    chosen_coords = random.choice(action_set_tuples)
    return game_engine_instance.coordinate_to_scalar(chosen_coords)

# --- Greedy Agent ---
def greedy_action(board: np.ndarray, valid_actions: list, policy_net: torch.nn.Module, device: torch.device,
                  env: HexEnv):
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

    return env.hex_game.scalar_to_coordinates(best_action_index)

# --- PPO Agent ---
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
        if valid_action_indices:  # Ensure there are valid actions
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
            chosen_action_tuple = random.choice(valid_actions) if valid_actions else env.hex_game.scalar_to_coordinates(
                0)  # Fallback
            action_coords = chosen_action_tuple
        else:
            dist = torch.distributions.Categorical(probs)
            action_scalar = dist.sample().item()
            action_coords = env.hex_game.scalar_to_coordinates(action_scalar)
        return action_coords


def save_model(agent, num_updates, win_rate, specific_agent="", wandb_logging_enabled=False):
    '''
    Save model with stats to given path
    '''
    model_path = os.path.join(MODEL_DIR, f"ppo_hex_agent_update_{num_updates}{specific_agent}_{win_rate}.pth")
    torch.save(agent.state_dict(), model_path)
    if wandb_logging_enabled:
        artifact = wandb.Artifact(f'model-{num_updates}', type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    # print(f"Model saved to {model_path}")

best_results_for_each_agent = {}

def update_best_agent_stats(win_rate: float, timesteps_collected: int, current_agent: str, agent_key: str,
                            agent=None) -> None:
    if agent_key not in best_results_for_each_agent:
        best_results_for_each_agent[agent_key] = {
            "win_rate": 0.0,
            "agent": None,
            "timesteps": 0,
            "updates": 0
        }

    if win_rate > best_results_for_each_agent[agent_key]["win_rate"]:
        best_results_for_each_agent[agent_key]["win_rate"] = round(win_rate, 2)
        best_results_for_each_agent[agent_key]["agent"] = current_agent
        best_results_for_each_agent[agent_key]["timesteps"] = timesteps_collected
        best_results_for_each_agent[agent_key]["updates"] += 1
        save_model(agent.policy, timesteps_collected, win_rate, best_results_for_each_agent[agent_key]["agent"])


def evaluate_mixed(agent, device, env: HexEnv, num_updates: int, frozen_agent, num_games=100,
                   wandb_logging_enabled=False):
    """
    Evaluate against:
    - greedy agent
    - ppo agent self
    - random agent
    - frozen self agent (updated every PERIODIC_FROZEN_AGENT_UPDATE_COUNTER updates)
    Collect results and output statistics
    """
    # assert len(frozen_agent_list) > 0, "At least one frozen agent .pth file is required for evaluation."
    print(
        f"\n--- Evaluating PPO Agent vs Random ({num_games}) + Frozen Agents ({len(frozen_agent_list)} total, {num_games} games) ---")

    stats = {
        Opponents.RANDOM: {"wins": 0, "games": 0, 'wins_as_white': 0, 'wins_as_black': 0, 'win_rate': 0.0},
        Opponents.FROZEN_SELF: {"wins": 0, "games": 0, 'wins_as_white': 0, 'wins_as_black': 0, 'win_rate': 0.0},
        Opponents.GREEDY: {"wins": 0, "games": 0, 'wins_as_white': 0, 'wins_as_black': 0, 'win_rate': 0.0},
        Opponents.SELF: {"wins": 0, "games": 0, 'wins_as_white': 0, 'wins_as_black': 0, 'win_rate': 0.0},
        'overall': {'wins': 0, 'games': num_games * (len(frozen_agent_list) + num_games)},
        'historical_comparison': {}
    }

    agent.policy.eval()
    game_engine = hexPosition(size=HEX_BOARD_SIZE)

    def play_game(opponent_policy_fn, stats_entry):
        for i in range(num_games):
            game_engine.reset()
            ppo_as_player = 1 if i % 2 == 0 else -1
            game_engine.player = 1

            while game_engine.winner == 0:
                current_board = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
                valid_actions = game_engine.get_action_space()

                if (game_engine.player == 1 and ppo_as_player == 1) or \
                        (game_engine.player == -1 and ppo_as_player == -1):
                    action = ppo_action_from_policy(current_board, valid_actions, agent.policy, device, env)
                else:
                    action = opponent_policy_fn(current_board, valid_actions)
                    if type(action) == int:
                        action = env.hex_game.scalar_to_coordinates(action)

                game_engine.move(action)
                game_engine.evaluate()

            stats_entry['games'] += 1
            if game_engine.winner == ppo_as_player:
                stats_entry['wins'] += 1
                stats['overall']['wins'] += 1
                if ppo_as_player == 1:
                    stats_entry['wins_as_white'] += 1
                else:
                    stats_entry['wins_as_black'] += 1

    # print("Evaluating against RANDOM...")
    play_game(lambda board, valid: random.choice(valid), stats[Opponents.RANDOM])

    # print("Evaluating against GREEDY...")
    play_game(lambda board, valid: greedy_action(board, valid, agent.policy, device, env),
              stats[Opponents.GREEDY])

    # print("Evaluating against SELF (mirror match)...")
    play_game(lambda board, valid: ppo_action_from_policy(board, valid, agent.policy, device, env),
              stats[Opponents.SELF])

    # --- Evaluate against frozen agents ---
    if len(frozen_agent_list) > 0:
        _, cur_frozen_agent = frozen_agent_list[1] if len(frozen_agent_list) == 3 else frozen_agent_list[0]
        for i in range(num_games):
            game_engine.reset()
            ppo_as_player = 1 if i % 2 == 0 else -1
            game_engine.player = 1

            while game_engine.winner == 0:
                current_board = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
                valid_actions = game_engine.get_action_space()

                if (game_engine.player == 1 and ppo_as_player == 1) or \
                        (game_engine.player == -1 and ppo_as_player == -1):
                    action = ppo_action_from_policy(current_board, valid_actions, agent.policy, device, env)
                else:
                    action = ppo_action_from_policy(current_board, valid_actions, cur_frozen_agent.policy, device, env)

                game_engine.move(action)
                game_engine.evaluate()

            stats[Opponents.FROZEN_SELF]['games'] += 1
            if game_engine.winner == ppo_as_player:
                stats[Opponents.FROZEN_SELF]['wins'] += 1
                stats['overall']['wins'] += 1
                if ppo_as_player == 1:
                    stats[Opponents.FROZEN_SELF]['wins_as_white'] += 1
                else:
                    stats[Opponents.FROZEN_SELF]['wins_as_black'] += 1

    header = f"{'Opponent':<20} {'WINS':<10} {'BLACK/WHITE Wins':<25} {'WIN RATE':<8}"
    print(header)
    print("-" * len(header))
    for opponent in [Opponents.RANDOM, Opponents.GREEDY, Opponents.SELF, Opponents.FROZEN_SELF]:
        if stats[opponent]:
            wins, games = stats[opponent]['wins'], stats[opponent]['games']
            wins_as_black = stats[opponent]['wins_as_black']
            wins_as_white = stats[opponent]['wins_as_white']

            win_rate = wins / max(1, games)
            stats[opponent]['win_rate'] = round(win_rate, 2)
            print(
                f"{opponent:<20} {wins:>2} / {games:<6} [BLACK: {wins_as_black:<2} / WHITE: {wins_as_white:<2}]".ljust(
                    45) +
                f"{win_rate:.2f}".rjust(10))

            update_best_agent_stats(win_rate, num_updates, opponent, opponent, agent)

    # weight the win rates according to how much agent was trained with it
    all_opponents = sum(opponent_counts.values())
    random_ratio = round(opponent_counts[Opponents.RANDOM] / all_opponents, 2)
    greedy_ratio = round(opponent_counts[Opponents.GREEDY] / all_opponents, 2)
    frozen_ratio = round(opponent_counts[Opponents.FROZEN_SELF] / all_opponents, 2)
    self_ratio = round(opponent_counts[Opponents.SELF] / all_opponents, 2)
    current_win_rate = round(stats[Opponents.FROZEN_SELF]['win_rate'] * frozen_ratio +
                             stats[Opponents.SELF]['win_rate'] * self_ratio +
                             stats[Opponents.GREEDY]['win_rate'] * greedy_ratio +
                             stats[Opponents.RANDOM]['win_rate'] * random_ratio,
                             2)

    current_win_rate_unweighted = round((stats[Opponents.FROZEN_SELF]['win_rate'] +  # *frozen_ratio +
                                         stats[Opponents.SELF]['win_rate'] +  # *self_ratio +
                                         stats[Opponents.GREEDY]['win_rate'] +  # *greedy_ratio +
                                         stats[Opponents.RANDOM]['win_rate']) / 4,  # *random_ratio,
                                        2)

    print(f"Current weighted win rate overall: {current_win_rate} | unweighted: {current_win_rate_unweighted} \n")
    print(f"Ratios are: random {stats[Opponents.RANDOM]['win_rate']} * {random_ratio}, "
          f"greedy {stats[Opponents.GREEDY]['win_rate']} * {greedy_ratio},"
          f" frozen {stats[Opponents.FROZEN_SELF]['win_rate']} * {frozen_ratio}"
          f", self {stats[Opponents.SELF]['win_rate']} * {self_ratio}")
    print("Best win rate overall: ", overall_best[-1])
    # print("Played opponents in this eval loop: ", opponent_counts)

    if not hasattr(evaluate_mixed, 'win_history'):
        evaluate_mixed.win_history = []
    evaluate_mixed.win_history.append(current_win_rate_unweighted)
    if len(evaluate_mixed.win_history) > 3:
        evaluate_mixed.win_history.pop(0)

    moving_avg = np.mean(evaluate_mixed.win_history)
    print(f"3-Eval Moving Avg: {moving_avg:.2f}")

    if wandb_logging_enabled:
        wandb.log({
            "eval/random_win_rate": stats[Opponents.RANDOM]['win_rate'],
            "eval/greedy_win_rate": stats[Opponents.GREEDY]['win_rate'],
            "eval/self_win_rate": stats[Opponents.SELF]['win_rate'],
            "eval/frozen_win_rate": stats[Opponents.FROZEN_SELF]['win_rate'],
            "eval/weighted_win_rate": current_win_rate,
            "eval/unweighted_win_rate": current_win_rate_unweighted,
            "eval/moving_avg_win_rate": moving_avg,
            "timesteps": num_updates
        })

    if current_win_rate_unweighted >= overall_best[-1][1]:
        overall_best[-1] = (num_updates, current_win_rate_unweighted)
        save_model(agent.policy, num_updates, overall_best[-1][1], "_overall", wandb_logging_enabled)

    if num_updates % PERIODIC_FROZEN_AGENT_UPDATE_COUNTER == 0:
        freeze_agent_and_reset_policy(frozen_agent, agent, env, device, num_updates)

    print("--- Evaluation Finished ---\n")
    agent.policy.train()
    return current_win_rate_unweighted, moving_avg, stats


def freeze_agent_and_reset_policy(frozen_agent, agent, env, device, num_updates):
    """"
        Freezes the agent and resets opponent policy, so that ppo agent can play against older versions of itself.
    """
    if len(frozen_agent_list) == MAX_LEN_FROZEN_AGENTS:  # check against latest x
        frozen_agent_list.pop(0)  # remove first

    frozen_agent_list.append((num_updates, copy.deepcopy(agent)))
    if len(frozen_agent_list) == 3:
        frozen_agent = frozen_agent_list[1][1]
    else:
        frozen_agent = frozen_agent_list[0][1]
    env.set_opponent_policy(lambda b, va: ppo_action_from_policy(b, va, frozen_agent.policy, device, env),
                            f"Frozen_Agent_{num_updates}")


def ppo_turn(agent, state, valid_actions, temperature):
    """
    Simpler version of getting the action from the game
    """
    action_scalar_ppo, log_prob_ppo, _ = agent.select_action(state, valid_actions, temperature)
    return action_scalar_ppo, log_prob_ppo


def determine_opponent(with_random: bool, with_greedy: bool, with_frozen: bool, total_timesteps: int,
                       current_win_rate: float):
    '''
    Based on which agents are enabled, how far training is and how well the agent currently performs, determine first the current
    opponent ratio and then get the opponent from the given range of opponents based on rand_val value.
    '''
    if with_random and with_greedy and with_frozen:
        rand_val, random_ratio, greedy_ratio, frozen_ratio, _ = get_ratios(total_timesteps, current_win_rate)
        return determine_random_self_frozen_greedy(rand_val, random_ratio, frozen_ratio, greedy_ratio)
    elif with_random and with_frozen:
        rand_val, random_ratio, _, frozen_ratio, _ = get_ratios(total_timesteps, current_win_rate)
        return determine_opponent_random_self_frozen(rand_val, random_ratio, frozen_ratio)
    elif with_random and with_greedy:
        rand_val, random_ratio, greedy_ratio, _, _ = get_ratios(total_timesteps, current_win_rate)
        return determine_opponent_random_greedy_self(rand_val, random_ratio, greedy_ratio)
    elif with_random:
        rand_val, random_ratio, _, _, _ = get_ratios(total_timesteps, current_win_rate)
        return determine_opponent_random_self(rand_val, random_ratio)
    elif with_greedy:
        rand_val, _, greedy_ratio, _, _ = get_ratios(total_timesteps, current_win_rate)
        return determine_opponent_greedy_self(rand_val, greedy_ratio)
    else:
        return get_opponent(Opponents.SELF)


def get_opponent(opponent):
    '''
    Helper function to set the opponent for the stats and then return it
    '''
    opponent_counts[opponent] += 1
    opponent_counts_per_eval_loop[opponent] += 1
    return opponent


def determine_opponent_random_self(rand_val: float, random_opponent_ratio: float):
    '''
    Either get random or self opponent
    '''
    if rand_val < random_opponent_ratio:
        return get_opponent(Opponents.RANDOM)
    else:
        return get_opponent(Opponents.SELF)


def determine_opponent_random_greedy_self(rand_val: float, random_ratio: float, greedy_ratio: float):
    '''
    Either return random, greedy or self opponent
    '''
    if rand_val < random_ratio:
        return get_opponent(Opponents.RANDOM)

    elif rand_val < (random_ratio + greedy_ratio):
        return get_opponent(Opponents.GREEDY)
    else:
        return get_opponent(Opponents.SELF)


def determine_opponent_greedy_self(rand_val: float, greedy_ratio: float):
    '''
    Either return greedy or self opponent
    '''
    if rand_val < greedy_ratio:
        return get_opponent(Opponents.GREEDY)
    else:
        return get_opponent(Opponents.SELF)


def determine_opponent_random_self_frozen(rand_val: float, random_opponent_ratio: float, frozen_self_ratio: float):
    '''
    Either return random, self or frozen self opponent
    '''
    if rand_val < random_opponent_ratio:
        return get_opponent(Opponents.RANDOM)

    elif rand_val < (random_opponent_ratio + frozen_self_ratio):
        return get_opponent(Opponents.FROZEN_SELF)

    else:
        return get_opponent(Opponents.SELF)


def determine_random_self_frozen_greedy(rand_val: float, random_opponent_ratio: float, frozen_self_ratio: float,
                                        greedy_ratio: float):
    '''
    Get either random, self, frozen or greedy opponent
    '''
    random_cutoff = random_opponent_ratio
    frozen_cutoff = random_cutoff + frozen_self_ratio
    greedy_cutoff = frozen_cutoff + greedy_ratio

    if rand_val < random_cutoff:
        return get_opponent(Opponents.RANDOM)

    elif rand_val < frozen_cutoff:
        return get_opponent(Opponents.FROZEN_SELF)

    elif rand_val < greedy_cutoff:
        return get_opponent(Opponents.GREEDY)

    else:
        return get_opponent(Opponents.SELF)


def get_device():
    '''
    Determine which device to use
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU device for training.")
    # device = torch.device("cpu")  # FORCE CPU
    print(f"Training will run on: {device}")
    return device


def get_scheduler(agent):
    '''
    Initialize scheduler
    '''
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        patience=50,  # iterations the scheduler waits until it reduces the LR
        factor=0.95,  # factor the LR gets multiplicated with
        min_lr=5e-5,  # min LR that will be kept as lower boundary
        threshold=0.02,
        mode='min'  # minimize loss, maximize reward
    )
    return plateau_scheduler  # warmup_scheduler, plateau_scheduler


def update_scheduler(plateau_scheduler, loss):
    plateau_scheduler.step(loss)


def get_temperature(total_timesteps_collected):
    """
        Measurement to support more entropy, i.e., more exploration
    """
    progress = total_timesteps_collected / MAX_TOTAL_TIMESTEPS
    return FINAL_TEMPERATURE + (TEMPERATURE - FINAL_TEMPERATURE) * np.exp(-5 * progress)


def get_entropy_coef(current_step, current_win_rate):
    '''
    Based on progress and current performance, support more or less entropy
    '''
    base_coef = ENTROPY_COEF_INITIAL - (ENTROPY_COEF_INITIAL - ENTROPY_COEF_FINAL) * (
            current_step / MAX_TOTAL_TIMESTEPS)

    # support more exploration if win_rate is low
    if current_win_rate < 0.5:
        return min(0.05, base_coef * 1.3)  # explore more

    elif current_win_rate > 0.7:
        return max(ENTROPY_COEF_FINAL, base_coef * 0.8)  # explore less

    return base_coef


def get_ratios(total_timesteps_collected: int, current_win_rate: float):
    '''
    Based on how far the trainina is and the current win rate, determine the opponent ratio
    '''
    progress = min(total_timesteps_collected / MAX_TOTAL_TIMESTEPS, 1.0)
    difficulty = get_opponent_difficulty(current_win_rate, progress)

    if difficulty == 'easy':
        random_ratio = RANDOM_OPPONENT_RATIO_EASY
        greedy_ratio = GREEDY_OPPONENT_RATIO_EASY
        self_ratio_multiplier = 0.75
        frozen_ratio_multiplier = 0.25

    elif difficulty == 'medium':
        random_ratio = RANDOM_OPPONENT_RATIO_MEDIUM
        greedy_ratio = GREEDY_OPPONENT_RATIO_MEDIUM
        self_ratio_multiplier = 0.6
        frozen_ratio_multiplier = 0.4

    elif difficulty == 'hard':
        random_ratio = RANDOM_OPPONENT_RATIO_HARD
        greedy_ratio = GREEDY_OPPONENT_RATIO_HARD
        self_ratio_multiplier = 0.5
        frozen_ratio_multiplier = 0.5

    remaining = max(0.0, 1.0 - random_ratio - greedy_ratio)
    self_ratio = self_ratio_multiplier * remaining
    frozen_ratio = frozen_ratio_multiplier * remaining
    rand_val = random.random()

    return rand_val, random_ratio, greedy_ratio, frozen_ratio, self_ratio


def get_opponent_difficulty(current_win_rate: float, progress: float) -> str:
    """
    Combines progress and performance to determine difficulty.
    """
    if progress < 0.33:
        if current_win_rate > 0.6:
            return 'medium'
        else:
            return 'easy'

    elif progress < 0.66:
        if current_win_rate > 0.7:
            return 'hard'
        elif current_win_rate > 0.5:
            return 'medium'
        else:
            return 'easy'

    else:
        if current_win_rate > 0.6:
            return 'hard'
        elif current_win_rate > 0.4:
            return 'medium'
        else:
            return 'easy'

def set_opponent_policy(opponent_type, env, frozen_agent, device, num_updates, agent):
    """
    Set opponent policy in hex env for evaluating against opponent policy.
    """
    if opponent_type == Opponents.RANDOM:
        env.set_opponent_policy(
            lambda b, va: env.hex_game.scalar_to_coordinates(
                random_opponent_action_logic(env.hex_game)),
            Opponents.RANDOM.capitalize())
    elif opponent_type == Opponents.FROZEN_SELF:
        env.set_opponent_policy(
            lambda b, va:
            ppo_action_from_policy(b, va, frozen_agent.policy, device,
                                   env),
            Opponents.FROZEN_SELF.capitalize() + f"_{num_updates}")
    elif opponent_type == Opponents.GREEDY:
        env.set_opponent_policy(
            lambda b, va: greedy_action(b, va, agent.policy, device, env),
            Opponents.GREEDY.capitalize())
    else:
        env.set_opponent_policy(
            lambda b, va: ppo_action_from_policy(b, va, agent.policy, device,
                                                 env),
            Opponents.SELF.capitalize())

def train(with_random, with_greedy, with_frozen, setting, player):
    wandb_logging_enabled = False
    secrets.set_key()
    if os.getenv('WANDB_API_KEY'):
        print("WAND logging enabled")
        wandb_logging_enabled = True
        wandb.init(
            project="hex-ppo",
            config={
                "setup": setting,
                "temperature": TEMPERATURE,
                "final_temperature": FINAL_TEMPERATURE,
                "hex_board_size": HEX_BOARD_SIZE,
                "initial_learning_rate": INITIAL_LEARNING_RATE,
                "gamma": GAMMA,
                "k_epochs": K_EPOCHS,
                "eps_clip": EPS_CLIP,
                "gae_lambda": GAE_LAMBDA,
                "entropy_coef_initial": ENTROPY_COEF_INITIAL,
                "entropy_coef_final": ENTROPY_COEF_FINAL,
                "max_clipping": MAX_CLIPPING,
                "value_coef": VALUE_COEF,
                "max_len_frozen_agents": MAX_LEN_FROZEN_AGENTS,
                "max_total_timesteps": MAX_TOTAL_TIMESTEPS,
                "timesteps_per_batch": TIMESTEPS_PER_BATCH,
                "updates_per_eval": UPDATES_PER_EVAL,
                "updates_per_save": UPDATES_PER_SAVE,
                "random_opponent_ratio_easy": RANDOM_OPPONENT_RATIO_EASY,
                "random_opponent_ratio_medium": RANDOM_OPPONENT_RATIO_MEDIUM,
                "random_opponent_ratio_hard": RANDOM_OPPONENT_RATIO_HARD,
                "greedy_opponent_ratio_easy": GREEDY_OPPONENT_RATIO_EASY,
                "greedy_opponent_ratio_medium": GREEDY_OPPONENT_RATIO_MEDIUM,
                "greedy_opponent_ratio_hard": GREEDY_OPPONENT_RATIO_HARD,
                "reward_enabled": hex_env.WITH_REWARD_SHAPING,
                "move_penalty": hex_env.WITH_MOVE_PENALTY,
                "with_greedy": with_greedy,
                "with_frozen": with_frozen,
                "with_random": with_random,
            }
        )

    # reset path to align saved models folder with settings path
    global MODEL_DIR, BEST_MODEL_PATH
    MODEL_DIR = setting
    BEST_MODEL_PATH = f"{MODEL_DIR}/ppo_hex_agent_update_best_so_far.pth"

    device = get_device()
    env = HexEnv(size=HEX_BOARD_SIZE)
    obs_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    agent = PPOAgent(obs_shape, action_space_size, INITIAL_LEARNING_RATE, GAMMA, K_EPOCHS, EPS_CLIP, GAE_LAMBDA, device,
                     MAX_CLIPPING,
                     VALUE_COEF,
                     torch.cuda.is_available())

    memory = RolloutMemory(device)

    frozen_agent = copy.deepcopy(agent)
    freeze_agent_and_reset_policy(frozen_agent, agent, env, device, 0)

    plateau_scheduler = get_scheduler(agent)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Starting PPO training for Hex for {MAX_TOTAL_TIMESTEPS} timesteps...")
    print("Training on #experiments Branch")
    print(f"Batch size: {TIMESTEPS_PER_BATCH}, Updates per batch: {K_EPOCHS}")

    if with_random:
        print(f"Playing against random opponent with initially {RANDOM_OPPONENT_RATIO_EASY * 100}% probability.")

    total_timesteps_collected = 0
    num_updates = 0
    all_episode_rewards = []
    state, info = env.reset()
    current_episode_reward_accumulator = 0.0
    best_moving_avg = 0.0
    current_win_rate = 0.0
    moving_avg = 0.0
    episode_count = 0
    episode_length_sum = 0
    overall_best[-1] = (0, 0.0)

    episode_wins = {
        Opponents.GREEDY: 0,
        Opponents.RANDOM: 0,
        Opponents.FROZEN_SELF: 0,
        Opponents.SELF: 0,
    }

    opponent_type = determine_opponent(with_random, with_greedy, with_frozen, total_timesteps_collected,
                                       current_win_rate)

    ppo_agent_player_id = player

    # was previously set, but changed to fixed assignment, better results
    # if opponent_type in [Opponents.RANDOM, Opponents.FROZEN_SELF, Opponents.GREEDY]:
    #    ppo_agent_player_id = random.choice([1, -1])
    set_opponent_policy(opponent_type, env, frozen_agent, device, num_updates, agent)

    while total_timesteps_collected < MAX_TOTAL_TIMESTEPS:

        # ---- Do update loop
        for _ in range(TIMESTEPS_PER_BATCH):
            valid_actions = info["valid_actions"]
            action_scalar_to_env = -1  # Placeholder

            is_ppo_turn_for_memory = False
            current_player_in_game = env.hex_game.player  # Player whose turn it is in hex_engine
            # print(f"[BEFORE STEP] Current player: {env.hex_game.player} - opponent {opponent_type}")

            # ---- player or opponent make a move
            if current_player_in_game == ppo_agent_player_id or opponent_type == Opponents.SELF:
                # if current_player_in_game == 1:
                # print("CURRENT PLAYER IS SELF")
                # if opponent_type == "self":
                # print("OPPONENT IS SELF")
                is_ppo_turn_for_memory = True
                temperature = get_temperature(total_timesteps_collected)
                action_scalar_to_env, log_prob_ppo = ppo_turn(agent, state, valid_actions, temperature)

            elif opponent_type == Opponents.FROZEN_SELF:
                # print("FROZEN SELF OPPONENT")
                # train against latest saved
                action_scalar_to_env = env.hex_game.coordinate_to_scalar(
                    ppo_action_from_policy(state, valid_actions, frozen_agent.policy, device,
                                           env))  # ignore warning, frozen agent gets initialized, if periodic self is initialized



            elif opponent_type == Opponents.RANDOM:
                action_scalar_to_env = random_opponent_action_logic(env.hex_game)

            elif opponent_type == Opponents.GREEDY:
                action_scalar_to_env = env.hex_game.coordinate_to_scalar(
                    greedy_action(state, valid_actions, agent.policy, device, env))  # returns scalar

            else:
                print("ERROR: This state should never be reached.")

            # ----- environment gets updated and step saved, if move was agents move
            # print("CURRENT OPPONENT TYPE: ", opponent_type)
            # print("PPO plays as ", ppo_agent_player_id)
            # print("CURRENT PLAYER IN GAME: ", current_player_in_game)
            next_state, step_reward, done, truncated, next_info = env.step(action_scalar_to_env, current_player_in_game,
                                                                           ppo_agent_player_id)
            # print(f"[STEP] Action taken: {action_scalar_to_env}, reward: {step_reward}, done: {done}, next_player: {env.hex_game.player}")
            # print("CURRENT PLAYER IN GAME AFTER STEP: ", env.hex_game.player)
            if is_ppo_turn_for_memory:  # Only add to memory if PPO made the move
                memory.add(state, action_scalar_to_env, log_prob_ppo, step_reward, done or truncated)

            current_episode_reward_accumulator += step_reward  # Accumulate for episode outcome logging

            state = next_state
            info = next_info
            total_timesteps_collected += 1

            # ---- game is through, update stats, reset and determine opponent for next episode
            if done or truncated:
                all_episode_rewards.append(current_episode_reward_accumulator)

                winner = env.hex_game.winner
                agent_won = (winner == ppo_agent_player_id)
                episode_count += 1
                episode_length_sum += env.hex_game.move_count

                if agent_won:
                    episode_wins[opponent_type] += 1

                # print( f"[Episode End] Reward: {current_episode_reward_accumulator:.2f}, Winner: {winner}, PPO Agent ID: {ppo_agent_player_id}, Opponent: {opponent_type}, Agent won: {agent_won}")

                state, info = env.reset()
                current_episode_reward_accumulator = 0.0
                # Determine opponent for the new episode
                opponent_type = determine_opponent(with_random, with_greedy, with_frozen,
                                                   total_timesteps_collected, current_win_rate)

                set_opponent_policy(opponent_type, env, frozen_agent, device, num_updates, agent)

                # below was used in the beginning, but more stable agent when player id did not switch and with trained white
                # if opponent_type in [Opponents.RANDOM, Opponents.FROZEN_SELF, Opponents.GREEDY]:
                #    ppo_agent_player_id = random.choice([1, -1])
                # print(f"NEXT: PPO plays as {ppo_agent_player_id} against {opponent_type}")

            if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
                break

            # print(f"Done step {total_timesteps_collected}, current opponent type: {opponent_type}")

        # ---- update training
        if len(memory.states) > 0:
            entropy_coef = get_entropy_coef(total_timesteps_collected, current_win_rate)
            # print("Entropy Coefficient: ", entropy_coef)
            p_loss, v_loss, ent, combined_loss = agent.update(memory, entropy_coef, 512)
            memory.clear_memory()
            num_updates += 1

            # all rewards mean
            avg_rewards = np.mean(all_episode_rewards)

            update_scheduler(plateau_scheduler,
                             combined_loss)  # if loss is used, set reduceonplateaulrscheduler to mode='min' else rewards to 'max'

            # --- log outputs every update
            if num_updates % 1 == 0:
                current_lr = agent.optimizer.param_groups[0]['lr']
                avg_ep_reward_str = ""
                avg_recent_ep_reward = 0.0
                if len(all_episode_rewards) > 0:
                    avg_ep_reward_str = f", Avg Ep Reward: {avg_rewards:.2f}"

                print(
                    f"Update {num_updates}, Timesteps: {total_timesteps_collected}, LR: {current_lr:.7f}{avg_ep_reward_str}")
                print(f"  Losses: Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Entropy: {ent:.4f}")
                if episode_count > 0:
                    avg_length = episode_length_sum / episode_count

                    # using the precalculated ratios might not give a full picture but suffices for now
                    greedy = round(
                        episode_wins[Opponents.GREEDY] / max(opponent_counts_per_eval_loop[Opponents.GREEDY], 1e-5), 2)
                    random_agent = round(
                        episode_wins[Opponents.RANDOM] / max(opponent_counts_per_eval_loop[Opponents.RANDOM], 1e-5), 2)
                    frozen = round(
                        episode_wins[Opponents.FROZEN_SELF] / max(opponent_counts_per_eval_loop[Opponents.FROZEN_SELF],
                                                                  1e-5), 2)
                    self_play = round(
                        episode_wins[Opponents.SELF] / max(opponent_counts_per_eval_loop[Opponents.SELF], 1e-5), 2)

                    # stats were weighted, but change to unweighted
                    """print(f"  "   
                          f"random {random_agent} * {round(random_ratio, 2)} "
                          f"| self {self_play} * {round(self_ratio, 2)}"
                          f"| frozen {frozen} * {round(frozen_ratio, 2)} "
                          f"| greedy {greedy} * {round(greedy_ratio, 2)} "
                          )"""

                    # win_rate = (greedy * round(greedy_ratio, 2) + random_agent * round(random_ratio, 2) + frozen * round(frozen_ratio, 2) + self_play * round(self_ratio, 2))*100
                    win_rate = sum(episode_wins.values()) / episode_count * 100

                    print(f"  Episodes Played: {episode_count}, "
                          # f"Complete Training Avg Reward: {avg_reward:.2f}, "
                          # f"Training Avg Length: {avg_length:.1f}, Weighted Training Win Rate: {win_rate:.1f}%")
                          f"Training Avg Length: {avg_length:.1f}, Training Win Rate: {win_rate:.1f}%")

                    """print(f"  Wins/Trained Ratio per agent: "
                        f"random {episode_wins[Opponents.RANDOM]} | {random_ratio}, "
                        f"self {episode_wins[Opponents.SELF]} | {self_ratio}, "
                        f"frozen self {episode_wins[Opponents.FROZEN_SELF]} | {frozen_ratio} "
                        f"greedy {episode_wins[Opponents.GREEDY]} | {greedy_ratio}, "
                    )"""

                    if wandb_logging_enabled:
                        wandb.log({
                            "train/policy_loss": p_loss,
                            "train/value_loss": v_loss,
                            "train/entropy": ent,
                            "train/learning_rate": current_lr,
                            "train/avg_episode_reward": avg_recent_ep_reward,
                            "train/avg_episode_length": avg_length,
                            "train/weighted_win_rate": win_rate,
                            "train/win_rate_vs_random": random_agent,
                            "train/win_rate_vs_greedy": greedy,
                            "train/win_rate_vs_frozen": frozen,
                            "train/win_rate_vs_self": self_play,
                            "timesteps": total_timesteps_collected,
                            "update": num_updates
                        })

                    print("  Played opponents: ", opponent_counts_per_eval_loop)
                    for key, _ in opponent_counts_per_eval_loop.items():
                        opponent_counts_per_eval_loop[key] = 0

                    episode_count = 0
                    episode_length_sum = 0
                    episode_wins = {
                        Opponents.GREEDY: 0,
                        Opponents.RANDOM: 0,
                        Opponents.FROZEN_SELF: 0,
                        Opponents.SELF: 0,
                    }

            # --- periodic evaluation
            if num_updates > 0 and num_updates % UPDATES_PER_EVAL == 0:

                current_win_rate, moving_avg, stats = evaluate_mixed(agent, device, env, num_updates, frozen_agent, 100,
                                                                     wandb_logging_enabled)
                for key, _ in opponent_counts.items():
                    opponent_counts[key] = 0
                if moving_avg > best_moving_avg + 0.02:  # win rate avg got better
                    best_moving_avg = moving_avg
                    print(f"Saving model best current")
                    torch.save(agent.policy.state_dict(), BEST_MODEL_PATH)

        if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
            break

    save_model(agent.policy, num_updates, "", "last_save", wandb_logging_enabled)
    if wandb_logging_enabled:
        wandb.finish()
    env.close()
    for key, item in best_results_for_each_agent.items():
        print(f"{key} Best: {item}")

    print("Collected updates: ", num_updates)
    print(overall_best)

    print("Training finished.")


if __name__ == '__main__':
    # All of these should be done WITH REWARD SHAPING and saved in wandb as far as I know for the paper

    train(with_random=False, with_greedy=False, with_frozen=False, player=1,  setting="against-self-only-white-no-reward-shaping")
    train(with_random=False, with_greedy=False, with_frozen=False, player=-1,  setting="against-self-only-black-no-reward-shaping")


    """train(with_random=False, with_greedy=True, with_frozen=False, player=1,
          setting="against-self-greedy-white-no-reward-shaping")
    train(with_random=False, with_greedy=True, with_frozen=False, player=-1,
          setting="against-self-greedy-black-no-reward-shaping")

    # I think these have not been done within wandb - we can check which ones we want to include in the paper 
    train(with_random=True, with_greedy=True, with_frozen=False, player=1,
          setting="against-self-random-greedy-white-no-reward-shaping")
    train(with_random=True, with_greedy=True, with_frozen=False, player=-1,
          setting="against-self-random-greedy-black-no-reward-shaping")

    train(with_random=True, with_greedy=True, with_frozen=True, player=1,
          setting="against-self-random-greedy-frozen-white-no-reward-shaping")
    train(with_random=True, with_greedy=True, with_frozen=True, player=-1,
          setting="against-self-random-greedy-frozen-black-no-reward-shaping")

    train(with_random=True, with_greedy=False, with_frozen=True, player=1, setting="against-self-random-frozen-white-no-reward-shaping")
    train(with_random=True, with_greedy=False, with_frozen=True, player=-1, setting="against-self-random-frozen-black-no-reward-shaping")

    train(with_random=True, with_greedy=False, with_frozen=False, player=1, setting="against-self-random-white-no-reward-shaping")
    train(with_random=True, with_greedy=False, with_frozen=False, player=-1, setting="against-self-random-black-no-reward-shaping")
    """
'''
class HexStrategicAgent:
    """
    It measures connectivity (chain size) and estimates how close
    the player is to forming a complete path across the board.
    """

    def __init__(self, board_size=7):
        """
        board_size: the size of the Hex board (e.g., 11).
        """
        self.board_size = board_size

    def select_action(self, board_state, player_id):
        """
        Selects the "best" legal move by simulating each move and then
        evaluating the resulting board with a more refined heuristic.

        board_state: 2D list/array holding the current board state:
                     0 = empty, 1 = player 1, 2 = player 2 (adapt if needed)
        player_id:   the agent's ID (1 or 2).

        Returns: (row, col) of the chosen move.
        """
        legal_moves = self.get_legal_moves(board_state)
        if not legal_moves:
            return None  # No legal moves available

        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            sim_board = self.simulate_move(board_state, move, player_id)
            score = self.evaluate_position(sim_board, player_id)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def get_legal_moves(self, board_state):
        """
        Returns all empty positions on the board (legal moves).
        """
        legal = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == 0:  # 0 means empty
                    legal.append((r, c))
        return legal

    def simulate_move(self, board_state, move, player_id):
        """
        Returns a copy of board_state where player_id has placed a stone at 'move'.
        """
        r, c = move
        new_board = [row[:] for row in board_state]  # Deep copy of board_state
        new_board[r][c] = player_id
        return new_board

    def evaluate_position(self, board_state, player_id):
        """
        Combines two factors:
          1) connectivity_score (chain size or largest connected group)
          2) shortest_path_length (distance from one side to the other)

        The final score is a positive measure of how good this position is for player_id.
        We directly add the connectivity score, and subtract (board_size - path_length),
        so that shorter paths yield a higher score.
        You can adjust merging these metrics depending on your preference.
        """
        connectivity_score = self._calculate_connectivity(board_state, player_id)
        path_length = self._shortest_path_length(board_state, player_id)

        # The smaller the path_length, the better. So we invert or subtract it.
        # For example, add (board_size - path_length) to the connectivity score.
        # If path_length is None (meaning unreachable), we treat it as the largest penalty.
        if path_length is None:
            path_length_score = -self.board_size  # penalize unreachable
        else:
            # Closer to 0 means closer to winning, so let's scale it.
            path_length_score = self.board_size - path_length

        return connectivity_score + path_length_score

    def _calculate_connectivity(self, board_state, player_id):
        """
        Counts how many stones of player_id are connected
        (or finds the size of the largest connected cluster).
        """
        visited = set()
        longest_chain = 0

        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_state[r][c] == player_id and (r, c) not in visited:
                    chain_size = self._dfs_chain_size(board_state, r, c, player_id, visited)
                    if chain_size > longest_chain:
                        longest_chain = chain_size
        return longest_chain

    def _dfs_chain_size(self, board_state, row, col, player_id, visited):
        """
        Depth-first search to count the size of the connected group
        starting at (row, col) for player_id.
        """
        stack = [(row, col)]
        chain_count = 0

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            chain_count += 1

            for nr, nc in self._get_neighbors(r, c):
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board_state[nr][nc] == player_id and (nr, nc) not in visited:
                        stack.append((nr, nc))

        return chain_count

    def _shortest_path_length(self, board_state, player_id):
        """
        Estimates the length of the shortest path from one side of the board
        to the other for player_id.

        For example, if player_id == 1, we consider top-to-bottom paths.
        If player_id == 2, we consider left-to-right paths.

        We treat the player's stones (player_id) and empty cells (0) as passable,
        and the opponent's stones as blocked.

        Returns: An integer for the length, or None if no path exists.
        """
        from collections import deque

        visited = set()
        queue = deque()

        # If player_id == 1, start from the top row
        # If player_id == 2, start from the left column
        if player_id == 1:
            # Enqueue all passable cells in the top row
            for col in range(self.board_size):
                if board_state[0][col] in (0, 1):
                    queue.append((0, col, 0))  # (row, col, distance)
                    visited.add((0, col))
        else:
            # player_id == 2
            for row in range(self.board_size):
                if board_state[row][0] in (0, 2):
                    queue.append((row, 0, 0))
                    visited.add((row, 0))

        # BFS
        while queue:
            r, c, dist = queue[0]
            queue.popleft()

            # Check if we've reached the opposite side
            if player_id == 1 and r == self.board_size - 1:
                return dist
            if player_id == 2 and c == self.board_size - 1:
                return dist

            # Explore neighbors
            for nr, nc in self._get_neighbors(r, c):
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if (nr, nc) not in visited:
                        # Passable if empty or belongs to player_id
                        if board_state[nr][nc] in (0, player_id):
                            visited.add((nr, nc))
                            queue.append((nr, nc, dist + 1))

        # No path found
        return None

    def _get_neighbors(self, row, col):
        """
        Returns up to six adjacent coordinates for a cell in a Hex grid.
        """
        neighbors = [
            (row - 1, col),  # up
            (row + 1, col),  # down
            (row, col - 1),  # left
            (row, col + 1),  # right
            (row - 1, col + 1),  # diagonal up-right
            (row + 1, col - 1),  # diagonal down-left
        ]
        return neighbors
'''