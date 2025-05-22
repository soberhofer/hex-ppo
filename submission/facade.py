# Here should be the necessary Python wrapper for your model, in the form of a callable agent.
# Please make sure that the agent does actually work with the provided Hex module.

from submission.ppo_agent_facade import ppo_agent_logic

def agent (board, action_set):
    """
    This function serves as the entry point for your trained PPO agent.
    It calls the ppo_agent_logic from ppo_agent_facade to select an action.
    """
    return ppo_agent_logic(board, action_set)
