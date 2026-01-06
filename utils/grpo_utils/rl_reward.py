import re

'''
Rewards scales to 0.0 to 1.0 in floating points for simple functions,
for penalizing functions ranges from -1.0 to 1.0 in floating points.
'''

def format_reward(completions, **kwargs):
    '''Binary reward function for thinking mode.'''
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def telederm_reward(completions, solution, **kwargs):
    '''
    Docstring for telederm_reward
    
    :param completions: Description
    :param solution: Description
    :param kwargs: Description
    '''

    return None
    

    