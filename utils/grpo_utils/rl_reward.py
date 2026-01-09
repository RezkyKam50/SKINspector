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

if __name__ == "__main__":

    completion='''
## Thinking

Okay, let's take a look at this skin lesion. The description says it's raised, erythematous, and has a central umbilication with crusting. That's interesting. The location is the periorbital region, which is a common spot for skin cancers, especially basal cell carcinoma. 

Now, let's think about the morphology. It's described as elevated and nodular, which fits with a skin cancer profile. The size is about 1.5 cm, which is a decent size for a suspicious lesion. The color is erythematous, and there's crusting and some scaling. Hmm, crusting makes me think of something more aggressive, like a squamous cell carcinoma. 

But wait, there's also that central umbilication. That's a bit unusual for a typical basal cell carcinoma. Usually, BCC has a pearly appearance with rolled borders. This one seems more inflamed and crusted. Maybe it's a variant of BCC, like a nodular one? Or could it be something else entirely? 

Let's consider the differentials. Basal cell carcinoma is the most likely diagnosis, but squamous cell carcinoma is also on the table. And keratosis pilaris seems less likely, given the crusting and central depression. 

Now, about the ABCDE criteria. The lesion is asymmetrical, has irregular borders, and is larger than 6mm. It's also changing in size and color, which is concerning. These criteria are often used for melanoma, but they can also apply to other skin cancers. 

I see that there are risk factors listed, like sun exposure, fair skin, and a history of sunburns. These are definitely important to consider, especially since they increase the risk of skin cancer. 

So, what's my conclusion? Based on the description, I'm leaning towards basal cell carcinoma, but I'm not completely ruling out squamous cell carcinoma or another type of skin lesion. It's important to get a biopsy to confirm the diagnosis. 

In terms of next steps, I would recommend a biopsy to get a definitive diagnosis. It's also important to keep an eye on the lesion and follow up regularly to see if it changes. And, of course, consider the risk factors and take steps to prevent future skin issues. 

Overall, this is a complex case, and a biopsy will provide the clarity needed to make an accurate diagnosis.

## Final Response

The clinical description of the skin lesion is detailed and provides several key features that are useful for evaluating the potential diagnosis. The lesion is raised, erythematous, and has a central umbilication with crusting, which is a notable combination of characteristics. The location on the periorbital region is a common site for skin cancers, such as basal cell carcinoma (BCC) and squamous cell carcinoma (SCC). 

The morphology of the lesion as described is consistent with a nodular lesion, which can be seen in both BCC and SCC. However, the presence of central umbilication is less typical for BCC, which usually has a pearly appearance with rolled borders. This feature, along with the crusting and scaling, could suggest a variant of BCC or even a different type of skin lesion. 

The ABCDE criteria for melanoma are mentioned, but they are more commonly used for pigmented lesions. However, they can also apply to non-pigmented lesions, such as basal cell carcinoma, especially if there is asymmetry, irregular borders, and a significant change in size or color. 

The differentials listed, including BCC, SCC, and keratosis pilaris, are appropriate. However, given the crusting and central umbilication, it might be worth considering other possibilities such as a pyogenic granuloma or an inflammatory lesion. 

In terms of risk factors, the patient's history of sun exposure, fair skin, and previous sunburns are all relevant and increase the risk of developing skin cancer. 

Given the complexity of the lesion and the need for a definitive diagnosis, a biopsy is recommended to confirm the diagnosis. It's also important to monitor the lesion for any changes and to follow up regularly. Additionally, addressing the risk factors, such as sun protection and skin care, is crucial in managing this condition. 

Overall, the clinical description is thorough, and the differentials are appropriate. However, the central umbilication and crusting warrant further investigation to rule out other potential diagnoses. A biopsy will provide the necessary clarity to make an accurate diagnosis.
    '''


logger.info(f"Think: {thinking_content}")
logger.info(f"Resp.: {final_response_content}")



    

    