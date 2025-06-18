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
TEMPERATURE = 1.0
FINAL_TEMPERATURE = 1.0
HEX_BOARD_SIZE = 7
INITIAL_LEARNING_RATE = 0.0003 # this is the learning rate up until linear warm up goes
GAMMA = 0.99
K_EPOCHS = 8
EPS_CLIP = 0.15
GAE_LAMBDA = 0.92           # bias in advantage estimates
ENTROPY_COEF_INITIAL = 0.03 # higher means more exploration in the beginning, gets reduced throughout training with each update in ppo agent
ENTROPY_COEF_FINAL = 0.002
MAX_CLIPPING = 0.25
VALUE_COEF = 0.4
MAX_PATIENCE = 10

MAX_TOTAL_TIMESTEPS = 1500000  # Total timesteps to train for
TIMESTEPS_PER_BATCH = 2048   # Timesteps to collect per batch before updating
UPDATES_PER_EVAL = 10        # Evaluate model every X updates (e.g., 50 updates * 2048 steps/update = ~100k steps)
UPDATES_PER_SAVE = 250       # Save model every X updates (e.g., 250 updates * 2048 steps/update = ~500k steps)
# LR Scheduler: step_size is now in terms of number of updates
#LR_SCHEDULER_STEP_SIZE = 50 # Decay LR every X updates (e.g. 50 updates)
#LR_SCHEDULER_GAMMA = 0.9    # Multiplicative factor of LR decay
WARMUP_EPOCHS = int(0.15 * MAX_TOTAL_TIMESTEPS) # 15% of overall total steps

RANDOM_OPPONENT_RATIO_EASY = 0.3 # Play against random opponent for this fraction of episodes in the beginning
RANDOM_OPPONENT_RATIO_MEDIUM = 0.15
RANDOM_OPPONENT_RATIO_HARD = 0.05
GREEDY_OPPONENT_RATIO_EASY = 0.1
GREEDY_OPPONENT_RATIO_MEDIUM = 0.15
GREEDY_OPPONENT_RATIO_HARD = 0.3


AVG_REWARD_WINDOW = 50
NUM_EVAL_GAMES = 200 # Number of games for periodic evaluation
MODEL_DIR = "./models"
BEST_MODEL_PATH = f"{MODEL_DIR}/ppo_hex_agent_update_best_so_far.pth"
PERIODIC_REPLACE_COUNTER = 75

overall_best = [0.0]
frozen_agent_list = []

class Opponents:
    RANDOM = "random"
    SELF = "self"
    FROZEN_SELF= "frozen_self"
    GREEDY="greedy"

opponent_counts = {
    Opponents.RANDOM: 0,
    Opponents.SELF: 0,
    Opponents.FROZEN_SELF: 0,
    Opponents.GREEDY: 0,
}

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


def save_model(agent, num_updates, win_rate, specific_agent = ""):
    model_path = os.path.join(MODEL_DIR, f"ppo_hex_agent_update_{num_updates}{specific_agent}_{win_rate}.pth")
    torch.save(agent.state_dict(), model_path)
    #print(f"Model saved to {model_path}")

best_results_for_each_agent = {}

def update_best_agent_stats(random_win_rate: float, timesteps_collected: int, current_agent: str, agent_key: str, agent = None) -> None:
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

def evaluate_mixed(agent, device, env: HexEnv, time_steps_collected: int, num_games=100):
    """
    Evaluate against 50 random games and 50 x agents in frozen_agents games against frozen agents
    """
    # assert len(frozen_agent_list) > 0, "At least one frozen agent .pth file is required for evaluation."
    assert len(overall_best) > 0, "test"
    print(f"\n--- Evaluating PPO Agent vs Random ({num_games}) + Frozen Agents ({len(frozen_agent_list)} total, {num_games} games) ---")

    stats = {
        Opponents.RANDOM: {"wins": 0, "games": 0, 'wins_as_white': 0, 'wins_as_black': 0, 'win_rate': 0.0},
        Opponents.FROZEN_SELF: [{"wins": 0, "games": 0, 'wins_as_white': 0, 'wins_as_black': 0, 'win_rate': 0.0} for _ in frozen_agent_list],
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

        # --- Evaluate against random ---

    #print("Evaluating against RANDOM...")
    play_game(lambda board, valid: random.choice(valid), stats[Opponents.RANDOM])
    # --- Evaluate against greedy ---
    #print("Evaluating against GREEDY...")
    play_game(lambda board, valid: greedy_action(board, valid, agent.policy, device, env),
              stats[Opponents.GREEDY])
    # --- Evaluate against self ---
    #print("Evaluating against SELF (mirror match)...")
    play_game(lambda board, valid: ppo_action_from_policy(board, valid, agent.policy, device, env),
              stats[Opponents.SELF])

    # --- Evaluate against frozen agents ---

    for idx, (update_step, frozen_agent) in enumerate(frozen_agent_list):
        #print(f"Evaluating against FROZEN agent @ step {update_step}...")
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
                    action = ppo_action_from_policy(current_board, valid_actions, frozen_agent.policy, device, env)

                game_engine.move(action)
                game_engine.evaluate()

            stats[Opponents.FROZEN_SELF][idx]['games'] += 1
            if game_engine.winner == ppo_as_player:
                stats[Opponents.FROZEN_SELF][idx]['wins'] += 1
                stats['overall']['wins'] += 1
                if ppo_as_player == 1:
                    stats[Opponents.FROZEN_SELF][idx]['wins_as_white'] += 1
                else:
                    stats[Opponents.FROZEN_SELF][idx]['wins_as_black'] += 1

    header = f"{'Opponent':<20} {'WINS':<10} {'BLACK/WHITE Wins':<25} {'WIN RATE':<8}"
    print(header)
    print("-" * len(header))
    for opponent in [Opponents.RANDOM, Opponents.GREEDY, Opponents.SELF]:
        wins, games = stats[opponent]['wins'], stats[opponent]['games']
        wins_as_black = stats[opponent]['wins_as_black']
        wins_as_white = stats[opponent]['wins_as_white']
        win_rate = wins / max(1, games)


        print(
            f"{opponent:<20} {wins:>2} / {games:<6} [BLACK: {wins_as_black:<2} / WHITE: {wins_as_white:<2}]".ljust(45) +
            f"{win_rate:.2f}".rjust(10))

        previous_rate = 0
        if stats[opponent]:
            previous_rate = stats[opponent]['win_rate']

        update_best_agent_stats(win_rate, time_steps_collected, opponent, opponent, agent)
        # print("PREV ", previous_rate)
        # print("NEW " , win_rate)
        stats[opponent]['win_rate'] = round(win_rate, 2)
        if opponent == Opponents.SELF and win_rate > previous_rate and num_games % PERIODIC_REPLACE_COUNTER == 0:
            # print(f"Replacing current frozen_self with better version at {time_steps_collected}")
            freeze_agent_and_reset_policy(agent, env, device, time_steps_collected)
            #print(len(frozen_agent_list))

    current_win_rate_frozen = 0
    count = 0
    for i, frozen_stat in enumerate(stats[Opponents.FROZEN_SELF]):
        wins, games = frozen_stat["wins"], frozen_stat["games"]
        win_rate = wins / max(1, games)
        update_step = frozen_agent_list[i][0]
        frozen_stat['win_rate'] = round(win_rate, 2)
        label = f"Frozen Agent {update_step}"
        print(f"{label:<20} {wins:>2} / {games:<6} [BLACK: {wins_as_black:<2} / WHITE: {wins_as_white:<2}]".ljust(45) +
              f"{win_rate:.2f}".rjust(10))
        current_win_rate_frozen += win_rate
        update_best_agent_stats(win_rate, time_steps_collected, f"{Opponents.FROZEN_SELF}{update_step}", update_step,
                                agent)
        count +=1

    # weight the win rates according to how much agent was trained with it
    all_opponents = sum(opponent_counts.values())
    random_ratio = round(opponent_counts[Opponents.RANDOM] / all_opponents, 2)
    greedy_ratio = round(opponent_counts[Opponents.GREEDY] / all_opponents, 2)
    frozen_ratio = round(opponent_counts[Opponents.FROZEN_SELF] / all_opponents, 2)
    self_ratio = round(opponent_counts[Opponents.SELF] / all_opponents, 2)

    current_win_rate = round((current_win_rate_frozen/count)*frozen_ratio +
                             stats[Opponents.SELF]['win_rate']*self_ratio +
                             stats[Opponents.GREEDY]['win_rate']*greedy_ratio +
                             stats[Opponents.RANDOM]['win_rate']*random_ratio,
                             2)

    print(f"Current weighted win rate overall: { current_win_rate} \n")
    print(f"Ratios are: random {stats[Opponents.RANDOM] } * {random_ratio}, greedy {stats[Opponents.GREEDY]} * {greedy_ratio},"
          f" frozen {current_win_rate_frozen/count} * {frozen_ratio}"
          f", self {stats[Opponents.SELF]} * {self_ratio}")
    print("Best weighted win rate overall: ", overall_best[0])
    print("Played opponents count so far: ", opponent_counts)

    if not hasattr(evaluate_mixed, 'win_history'):
        evaluate_mixed.win_history = []
    evaluate_mixed.win_history.append(current_win_rate)
    if len(evaluate_mixed.win_history) > 3:
        evaluate_mixed.win_history.pop(0)

    moving_avg = np.mean(evaluate_mixed.win_history)
    print(f"3-Eval Moving Avg: {moving_avg:.2f}")
    if current_win_rate > overall_best[0]:
        overall_best[0] = current_win_rate
        save_model(agent.policy, time_steps_collected, overall_best[0], "_overall")

    print("--- Evaluation Finished ---\n")
    agent.policy.train()
    return current_win_rate, moving_avg, stats

def freeze_agent_and_reset_policy(agent, env, device, num_updates):
    """"
        Freezes the agent and resets the policy, so that ppo agent can play against older versions of itself.
    """
    frozen_agent = copy.deepcopy(agent)

    # TODO: check if this can lead to memory problems
    if len(frozen_agent_list) == 3: # check against latest three
        frozen_agent_list.pop(0)    # remove first

    frozen_agent_list.append((num_updates, frozen_agent))

    env.set_opponent_policy(lambda b, va: ppo_action_from_policy(b, va, frozen_agent.policy, device, env))
    #print(f"Opponent replaced with frozen snapshot at update {num_updates}")

def ppo_turn(agent, state, valid_actions, temperature):
    action_scalar_ppo, log_prob_ppo, _ = agent.select_action(state, valid_actions, temperature)
    return action_scalar_ppo, log_prob_ppo

def determine_opponent(with_random: bool, with_periodic_self: bool, rand_val: float, random_opponent_ratio: float, frozen_self_ratio: float,  greedy_ratio: float, self_ratio: float, force_first = False):
    if force_first:
        return Opponents.FROZEN_SELF

    random_cutoff = random_opponent_ratio
    frozen_cutoff = random_cutoff + frozen_self_ratio
    greedy_cutoff = frozen_cutoff + greedy_ratio
    #print(rand_val, random_ratio, greedy_ratio, self_ratio, frozen_ratio)

    #print(rand_val, random_opponent_ratio, frozen_self_ratio, greedy_ratio, self_ratio)

    if with_random and rand_val < random_cutoff:
        #print("CHOOSE RANDOM")
        opponent_counts[Opponents.RANDOM] += 1
        return Opponents.RANDOM
    elif with_periodic_self and rand_val < frozen_cutoff:
        #print("CHOOSE PERIODIC")
        opponent_counts[Opponents.FROZEN_SELF] += 1
        return Opponents.FROZEN_SELF
    elif rand_val < greedy_cutoff:
        #print("CHOOSE GREEDY")
        opponent_counts[Opponents.GREEDY] += 1
        return Opponents.GREEDY
    else:
        #print("CHOOSE SELF")
        opponent_counts[Opponents.SELF] += 1
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
    """warmup_scheduler = torch.optim.lr_scheduler.LinearLR(agent.optimizer,
                                                         start_factor=0.1,
                                                         total_iters=WARMUP_EPOCHS
                                                         # iterations until which the initial LR is reached
                                                         )"""

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        patience=15,  # iterations the scheduler waits until it reduces the LR
        factor=0.9,  # factor the LR gets multiplicated with
        min_lr=5e-5,  # min LR that will be kept as lower boundary
        threshold=0.02,
        mode='min'   # minimize loss, maximize reward
    )
    return plateau_scheduler #  warmup_scheduler, plateau_scheduler

#warmup_scheduler,
def update_scheduler(num_updates, plateau_scheduler, loss, avg_reward):
    #if num_updates < WARMUP_EPOCHS:
    #    warmup_scheduler.step()
    #else:
        # plateau scheduler needs step
        plateau_scheduler.step(loss)
    # lr_scheduler.step()

def get_temperature(total_timesteps_collected):
    """
        Measurement to support more entropy, i.e., more exploration --> agent as is gets stuck too fast
    """
    progress = total_timesteps_collected / MAX_TOTAL_TIMESTEPS
    return FINAL_TEMPERATURE + (TEMPERATURE - FINAL_TEMPERATURE) * np.exp(-5 * progress)


def get_entropy_coef(current_step, current_win_rate):
    base_coef = ENTROPY_COEF_INITIAL - (ENTROPY_COEF_INITIAL - ENTROPY_COEF_FINAL) * (
                current_step / MAX_TOTAL_TIMESTEPS)

    # support more exploration if win_rate is low
    if current_win_rate < 0.5:
        return min(0.05, base_coef * 1.3)  # explore more
    elif current_win_rate > 0.7:
        return max(ENTROPY_COEF_FINAL, base_coef * 0.8)  # explore less
    return base_coef ## else return coef based on progress


def get_ratios(num_updates: int, current_win_rate: float):
    if current_win_rate < 0.4:
        random_ratio = RANDOM_OPPONENT_RATIO_EASY
        greedy_ratio = GREEDY_OPPONENT_RATIO_EASY
        if len(frozen_agent_list) >= 1:
            self_ratio_multiplier = 0.8
            frozen_ratio_multiplier = 0.2
        else:
            self_ratio_multiplier = 1.0
            frozen_ratio_multiplier = 0.0

    elif current_win_rate < 0.7:
        random_ratio = RANDOM_OPPONENT_RATIO_MEDIUM
        greedy_ratio = GREEDY_OPPONENT_RATIO_MEDIUM
        if len(frozen_agent_list) >= 1:
            self_ratio_multiplier = 0.85
            frozen_ratio_multiplier = 0.15
        else:
            self_ratio_multiplier = 1.0
            frozen_ratio_multiplier = 0.0

    else:
        random_ratio = RANDOM_OPPONENT_RATIO_HARD
        greedy_ratio = GREEDY_OPPONENT_RATIO_HARD
        if len(frozen_agent_list) >= 1:
            self_ratio_multiplier = 0.9
            frozen_ratio_multiplier = 0.1
        else:
            self_ratio_multiplier = 1.0
            frozen_ratio_multiplier = 0.0

    remaining = max(0.0, 1.0 - random_ratio - greedy_ratio)
    self_ratio = self_ratio_multiplier * remaining
    frozen_ratio = frozen_ratio_multiplier * remaining
    # frozen_ratio = 0.0

    rand_val = random.random()
    return rand_val, random_ratio, greedy_ratio, frozen_ratio, self_ratio

def train(with_periodic_self: bool = True, with_random: bool = True):
    device = get_device()
    env = HexEnv(size=HEX_BOARD_SIZE)
    obs_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    agent = PPOAgent(obs_shape, action_space_size, INITIAL_LEARNING_RATE, GAMMA, K_EPOCHS, EPS_CLIP, GAE_LAMBDA, device, MAX_CLIPPING,
                     VALUE_COEF,
                     torch.cuda.is_available())

    memory = RolloutMemory(device)
    # frozen agent probability set to 0 as long as no agent is saved
    frozen_agent = None


    if with_periodic_self:
        frozen_agent = copy.deepcopy(agent)
        # try self play instead of HEXGAME agent
        def self_play_opponent(board, valid_actions):
            return ppo_action_from_policy(board, valid_actions, agent.policy, device, env)

        env.set_opponent_policy(self_play_opponent)
        freeze_agent_and_reset_policy(agent, env, device, 0)

    #lr_scheduler = StepLR(agent.optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    #warmup_scheduler,
    plateau_scheduler = get_scheduler(agent)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Starting PPO training for Hex for {MAX_TOTAL_TIMESTEPS} timesteps...")
    print("Training on #experiments Branch")
    print(f"Batch size: {TIMESTEPS_PER_BATCH}, Updates per batch: {K_EPOCHS}")

    if with_random:
        print(f"Playing against random opponent with initially {RANDOM_OPPONENT_RATIO_EASY*100}% probability.")

    total_timesteps_collected = 0
    num_updates = 0
    all_episode_rewards = []
    state, info = env.reset()
    current_episode_reward_accumulator = 0.0
    best_moving_avg = 0.0
    patience_counter = 0
    current_win_rate = 0.0
    episode_count = 0
    episode_reward_sum = 0.0
    episode_length_sum = 0
    episode_wins = 0

    rand_val, random_ratio, greedy_ratio, frozen_ratio, self_ratio = get_ratios(num_updates, current_win_rate)
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
            # print("CURRENT OPPONENT TYPE: ", opponent_type)
            next_state, step_reward, done, truncated, next_info = env.step(action_scalar_to_env, ppo_agent_player_id)
            #print(f"[STEP] Action taken: {action_scalar_to_env}, reward: {step_reward}, done: {done}, next_player: {env.hex_game.player}")

            if is_ppo_turn_for_memory: # Only add to memory if PPO made the move
                memory.add(state, action_scalar_to_env, log_prob_ppo, step_reward, done or truncated)

            current_episode_reward_accumulator += step_reward # Accumulate for episode outcome logging

            state = next_state
            info = next_info
            total_timesteps_collected += 1

            # ---- game is through, update stats, reset and determine opponent for next episode
            if done or truncated:
                all_episode_rewards.append(current_episode_reward_accumulator)

                winner = env.hex_game.winner
                agent_won = (winner == ppo_agent_player_id)
                episode_count +=1
                episode_reward_sum += current_episode_reward_accumulator
                episode_length_sum += env.hex_game.move_count
                if agent_won:
                    episode_wins += 1
                #print( f"[Episode End] Reward: {current_episode_reward_accumulator:.2f}, Length: {episode_length}, Winner: {winner}, PPO Agent ID: {ppo_agent_player_id}, Opponent: {opponent_type}, Agent won: {agent_won}")

                state, info = env.reset()
                current_episode_reward_accumulator = 0.0
                # Determine opponent for the new episode
                rand_val, random_ratio, greedy_ratio, frozen_ratio, self_ratio = get_ratios(num_updates, current_win_rate)
                opponent_type = determine_opponent(with_random, with_periodic_self, rand_val, random_ratio, frozen_ratio, greedy_ratio, self_ratio)

                if opponent_type in [Opponents.RANDOM, Opponents.FROZEN_SELF, Opponents.GREEDY]:
                    ppo_agent_player_id = random.choice([1, -1])

            if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
                break

            #rint(f"Done step {total_timesteps_collected}, current opponent type: {opponent_type}")

        # ---- update training
        if len(memory.states) > 0:
            entropy_coef = get_entropy_coef(total_timesteps_collected, current_win_rate)
            #print("Entropy Coefficient: ", entropy_coef)
            p_loss, v_loss, ent, combined_loss = agent.update(memory, entropy_coef, 512)
            memory.clear_memory()
            num_updates += 1

            # LR updates --> correct scheduler should get increased
            # update_scheduler(num_updates, warmup_scheduler, plateau_scheduler, v_loss)
            # last x rewards mean
            avg_rewards = np.mean(all_episode_rewards[-AVG_REWARD_WINDOW:])

            # try to update scheduler based on increasing rewards
            update_scheduler(num_updates, #warmup_scheduler
                              plateau_scheduler, combined_loss,
                             avg_rewards)  # if loss is used, set reduceonplateaulrscheduler to mode='min'

            # --- log outputs every ten updates
            if num_updates % 1 == 0:
                current_lr = agent.optimizer.param_groups[0]['lr']
                avg_ep_reward_str = ""
                if len(all_episode_rewards) > 0:
                    lookback_episodes = min(AVG_REWARD_WINDOW, len(all_episode_rewards))
                    avg_recent_ep_reward = np.mean(all_episode_rewards[-lookback_episodes:])
                    avg_ep_reward_str = f", Avg Ep Reward (last ~{lookback_episodes}): {avg_recent_ep_reward:.2f}"

                print(f"Update {num_updates}, Timesteps: {total_timesteps_collected}, LR: {current_lr:.7f}{avg_ep_reward_str}")
                print(f"  Losses: Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Entropy: {ent:.4f}")
                if episode_count > 0:
                    avg_reward = episode_reward_sum / episode_count
                    avg_length = episode_length_sum / episode_count
                    win_rate = episode_wins / episode_count * 100

                    print(f"  Episodes Played: {episode_count}, Training Avg Reward: {avg_reward:.2f}, Training Avg Length: {avg_length:.1f}, Training Win Rate: {win_rate:.1f}%")

                    episode_count = 0
                    episode_reward_sum = 0.0
                    episode_length_sum = 0
                    episode_wins = 0

            # --- periodic evaluation
            if num_updates > 0 and num_updates % UPDATES_PER_EVAL == 0:

                current_win_rate, moving_avg, stats = evaluate_mixed(agent, device, env, num_updates)
                if moving_avg > best_moving_avg + 0.02:  # win rate avg got better
                    best_moving_avg = moving_avg
                    patience_counter = 0
                    torch.save(agent.policy.state_dict(), BEST_MODEL_PATH)


                else:                                   # win rate avg stagnates or gets worse
                    patience_counter += 1
                    if patience_counter >= MAX_PATIENCE:
                        print(f"No improvement for {MAX_PATIENCE} evaluations. Restoring best model.")
                        # Load best model
                        agent.policy.load_state_dict(torch.load(BEST_MODEL_PATH))
                        # reduce learning rate hard
                        for g in agent.optimizer.param_groups:
                            g['lr'] *= 0.9
                        patience_counter = 0

                #evaluate_against_random(agent.policy, device, env, NUM_EVAL_GAMES)

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


# --- Evaluation Function (integrated) ---
def evaluate_against_random(ppo_policy_net, device, env: HexEnv, num_games=NUM_EVAL_GAMES, ):
    print(f"\n--- Evaluating PPO Agent vs Random Agent for {num_games} games ---")
    ppo_wins = 0
    game_engine = hexPosition(size=HEX_BOARD_SIZE)
    ppo_policy_net.eval()  # Ensure ppo_policy_net is in eval mode

    for i in range(num_games):
        game_engine.reset()
        if i % 2 == 0:
            current_player1_is_ppo = True
            ppo_plays_as_player = 1  # PPO is player 1 (White)
        else:
            current_player1_is_ppo = False
            ppo_plays_as_player = -1  # PPO is player -1 (Black)

        while game_engine.winner == 0:

            # convert board to tensor, add batch & channel dimensions
            current_board_for_nn = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            action_coords = None

            is_ppo_turn_now = (game_engine.player == 1 and current_player1_is_ppo) or \
                              (game_engine.player == -1 and not current_player1_is_ppo)

            if is_ppo_turn_now:
                with torch.no_grad():
                    valid_actions_tuples = game_engine.get_action_space()
                    action_coords = ppo_action_from_policy(current_board_for_nn, valid_actions_tuples, ppo_policy_net,
                                                           device, env)

            else:  # Random agent's turn
                action_coords = random.choice(game_engine.get_action_space())  # random_agent_eval simplified
                # action_coords = random_agent_eval(game_engine.board, game_engine.get_action_space())

            if action_coords is None:  # Should not happen if logic is correct
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
    ppo_policy_net.train()  # Set policy back to train mode
    return win_rate
