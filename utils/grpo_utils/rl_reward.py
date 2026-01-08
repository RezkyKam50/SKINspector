import re
from loguru import logger
from reward_llm import Judge

'''
Rewards scales to 0.0 to 1.0 in floating points for simple functions,
for penalizing functions ranges from -1.0 to 1.0 in floating points.
'''

def format_reward(completions, **kwargs):
    '''Binary reward function for thinking mode.
    '''

    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def style_reward(completions, **kwargs):
    '''Style reward with LLM-as a judge
    '''

    fake_image = "./examples/testing/basal_1.jpg"
    factor = {
        "TECH_S"    : 0.25,
        "WRT_S"     : 0.10,
        "CLIN_RS"   : 0.20,
        "SFTY_S"    : 0.15,
        "CPLT_S"    : 0.10,
        "SPEC_S"    : 0.05,
        "EDU_S"     : 0.05,
        "LIM_S"     : 0.05,
        "STRC_S"    : 0.03,
        "ACT_S"     : 0.02
    }

    result = Judge(completions, image_path, fake_image, factor=factor) # TODO: get image_path from dataset
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*80}\n")
    
    # Core scores
    logger.info(f"Overall Score: {result['score']:.3f}")
    logger.info(f"\n{'-'*80}")
    logger.info("DIMENSION SCORES:")
    logger.info(f"{'-'*80}")
    logger.info(f"Technical Accuracy:           {result['technical_accuracy']:.3f}")
    logger.info(f"Writing Style:                {result['writing_style']:.3f}")
    logger.info(f"Clinical Reasoning:           {result['clinical_reasoning']:.3f}")
    logger.info(f"Safety & Urgency:             {result['safety_urgency']:.3f}")
    logger.info(f"Completeness:                 {result['completeness']:.3f}")
    logger.info(f"Diagnostic Specificity:       {result['diagnostic_specificity']:.3f}")
    logger.info(f"Educational Value:            {result['educational_value']:.3f}")
    logger.info(f"Accuracy of Limitations:      {result['accuracy_limitations']:.3f}")
    logger.info(f"Structured Presentation:      {result['structured_presentation']:.3f}")
    logger.info(f"Clinical Actionability:       {result['clinical_actionability']:.3f}")
    
    logger.info(f"\n{'-'*80}")
    logger.info("DETAILED REASONING:")
    logger.info(f"{'-'*80}")
    logger.info(f"\n{result['reasoning']}\n")
    logger.info(f"{'='*80}\n")

    return None

def terminologies_reward(completions, solution, **kwargs):
    '''
    Docstring for telederm_reward
    
    :param completions: Description
    :param solution: Description
    :param kwargs: Description
    '''

    return None

def criteria_reasoning_reward(completion, solution, debug=None, **kwargs):
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)


    if not think_match or not answer_match:
        return 0.0
    
    thinking = think_match.group(1)
    answer = answer_match.group(1)
    
    if debug:
        logger.info(thinking)
        logger.info(answer)

    reward = 0.0
    
    diagnosis_terms = ['differential', 'consider', 'rule out', 'likely', 'possible']
    if any(term in thinking.lower() for term in diagnosis_terms):
        reward += 0.2
     
    features = ['border', 'color', 'size', 'texture', 'symmetry']
    feature_count = sum(1 for f in features if f in thinking.lower())
    reward += 0.2 * (feature_count / len(features))
     
    urgency_terms = ['urgent', 'biopsy', 'refer', 'monitor', 'follow-up']
    if any(term in answer.lower() for term in urgency_terms):
        reward += 0.2
    
    return reward

# if __name__ == "__main__":

#     completion='''
#     <think>
#     The lesion exhibits asymmetry, irregular borders, and color 
#     variation. Considering differential diagnosis: melanoma vs 
#     dysplastic nevus vs seborrheic keratosis.
#     </think>
#     <answer>
#     Most likely diagnosis: Melanoma. The ABCDE criteria are 
#     concerning. Recommend urgent dermatology referral and biopsy.
#     </answer>
#     '''

#     logger.info(f"Reward: {criteria_reasoning_reward(completion, None, debug=True)}")



    

    