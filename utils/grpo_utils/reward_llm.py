import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from rl_prompt import *
from loguru import logger
import re
import base64
import time
import sys

_CPU_CONF = {
    'threads': 6
}

_GPU_CONF = {
    'gpu_layers': 99,
    'flash_attention': True
}


class Gemma3ChatHandler(Llava15ChatHandler):
    # borrowed from: https://github.com/abetlen/llama-cpp-python/pull/1989
    # Chat Format:
    # '<bos><start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n'

    DEFAULT_SYSTEM_MESSAGE = None

    CHAT_FORMAT = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% if messages[0]['content'] is string %}"
        "{% set first_user_prefix = messages[0]['content'] + '\n\n' %}"
        "{% else %}"
        "{% set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' %}"
        "{% endif %}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{% set first_user_prefix = \"\" %}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{% endif %}"
        "{% if (message['role'] == 'assistant') %}"
        "{% set role = \"model\" %}"
        "{% else %}"
        "{% set role = message['role'] %}"
        "{% endif %}"
        "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] | trim }}"
        "{% elif message['content'] is iterable %}"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image_url' and item['image_url'] is string %}"
        "{{ '\n\n' + item['image_url'] + '\n\n' }}"
        "{% elif item['type'] == 'image_url' and item['image_url'] is mapping %}"
        "{{ '\n\n' + item['image_url']['url'] + '\n\n' }}"
        "{% elif item['type'] == 'text' %}"
        "{{ item['text'] | trim }}"
        "{% endif %}"
        "{% endfor %}"
        "{% else %}"
        "{{ raise_exception(\"Invalid content type\") }}"
        "{% endif %}"
        "{{ '<end_of_turn>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\n' }}"
        "{% endif %}"
    )

def image_to_base64_data_uri(file_path: str, mime_type: str = "image/jpeg") -> str:
    logger.info("Encoding image...")
    start = time.time()
    with open(file_path, "rb") as img_file:
        base64_data = base64.encodebytes(img_file.read()).decode("ascii")
    logger.info(f"Encoding completed in {time.time() - start:.6f}s")
    return f"data:{mime_type};base64,{base64_data}"


def MedGemmaExpert(path_to_image: str, gpu: bool) -> str:
    logger.info("Generating reference description...")
    image_uri = image_to_base64_data_uri(path_to_image)
    chat_handler = Gemma3ChatHandler(
        clip_model_path="./models/MedGemma/mmproj-F16.gguf", 
        verbose=False
    )
    llm = Llama(
        model_path="./models/MedGemma/medgemma-4b-it-Q4_K_M.gguf",
        n_ctx=4096,
        use_mmap=True,
        n_threads=_CPU_CONF["threads"],
        n_gpu_layers=_GPU_CONF['gpu_layers'] if gpu else 0,
        flash_attn=_GPU_CONF["flash_attention"] if gpu else False,
        chat_handler=chat_handler,
        verbose=False
    )
    
    start = time.time()
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """You are an expert dermatologist. Describe skin lesions using:

                1. Morphology: lesion type, size, shape, color, borders, texture
                2. Location and distribution
                3. Key clinical features
                4. Most likely diagnosis with 2-3 differentials
                5. Risk factors (ABCDE criteria if pigmented)

                Use precise medical terminology. Be concise but comprehensive.

                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": "Provide a clinical description of this dermatological finding."}
                ]
            }
        ],
        max_tokens=1024,
        temperature=0.0,
        top_p=0.95
    )

    result = output['choices'][0]['message']['content']
    logger.info(f"Reference generated in {time.time() - start:.2f}s")
    logger.info(f"MedGemma - {result}")

    return output['choices'][0]['message']['content']


def HuaTuoExpert(reference: str, critic_mode: bool,gpu: bool) -> dict[str, any]:
    logger.info("Evaluating MedGemma diagnosis...")
    llm = Llama(
        model_path="./models/HuaTuo/HuatuoGPT-o1-8B.gguf",
        chat_format="llama-3",
        n_ctx=2048,
        use_mmap=True,
        n_threads=_CPU_CONF["threads"],
        n_gpu_layers=_GPU_CONF['gpu_layers'] if gpu else 0,
        flash_attn=_GPU_CONF["flash_attention"] if gpu else False,
        verbose=False
    )
    
    start = time.time()

    if reference is None:
        reference = "Candidate didn't give a respond."

    if critic_mode:
        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a rigorous senior dermatologist conducting a critical peer review. Your role is to identify weaknesses, gaps, and potential errors in clinical assessments.
                    
                    Provide a thorough critique focusing on:

                    1. **Missing Information**: What critical clinical details are absent? (e.g., lesion size, duration, patient history, location specifics)
                    2. **Diagnostic Weaknesses**: Is the primary diagnosis adequately justified? What evidence is lacking or contradictory?
                    3. **Overlooked Differentials**: What alternative diagnoses should have been considered but weren't? Are there red flags being missed?
                    4. **Incomplete Risk Assessment**: What risk factors or warning signs (ABCDE criteria, malignancy indicators) were inadequately addressed?
                    5. **Insufficient Clinical Recommendations**: Are the proposed next steps appropriate? What critical actions are missing (biopsy, urgent referral, specific imaging)?
                    6. **Clinical Reasoning Gaps**: Where is the logical reasoning flawed or incomplete?
                    
                    **Important**: The assessment is based solely on clinical description without imaging confirmation, which introduces significant diagnostic uncertainty.
                    
                    Be direct and specific about deficiencies. If the assessment is strong, acknowledge it, but prioritize identifying areas for improvement and potential clinical risks.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Critically evaluate this dermatological assessment and identify its weaknesses:

                    "{reference}"

                    Focus on what's missing, inadequate, or potentially problematic in this clinical reasoning.
                    """
                }
            ],
            max_tokens=2048,
            temperature=1.2,
            top_p=0.95
        )
    else:
        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a senior dermatologist providing a critical second opinion. 
                    You will review and provide:

                    1. Clinical validation: Assess the accuracy and completeness of the morphological description
                    2. Diagnostic reasoning: Evaluate whether the primary diagnosis is well-supported by the described features
                    3. Differential refinement: Comment on whether the differentials are appropriate, and suggest additions if warranted
                    4. Risk assessment: Validate or expand on risk stratification (e.g., ABCDE criteria for melanocytic lesions)
                    5. Clinical recommendations: Suggest next steps (biopsy, imaging, follow-up, reassurance)
                    6. The assessment doesn't provide an image which means the evidence is only based on the clinical description.
                    
                    Be precise, evidence-based, and constructive. Highlight both strengths, weakness and critical gaps (if present) in the assessment.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Please review the following dermatological assessment:

                    "{reference}"

                    Provide your second opinion on this clinical description and diagnostic reasoning.
                    """
                }
            ],
            max_tokens=2048,
            temperature=0.4,
            top_p=0.95
        )
    
    result = output['choices'][0]['message']['content']
    logger.info(f"Validation completed in {time.time() - start:.2f}s")
    logger.info(f"HuaTuo - {result}")

    t = re.search(r'## Thinking\n+(.*?)(?=## Final Response|$)', result, re.DOTALL)
    t_content = t.group(1).strip() if t else None
    r = re.search(r'## Final Response\n+(.*)', result, re.DOTALL)
    r_content = r.group(1).strip() if r else None
 
    return {
        'full': result, 
        'reasoning': t_content, 
        'answer': r_content
    }

def MedGemmaScorer(reference: list, candidate: str, path_to_image: str, path_to_fw_image:str, factor:dict, gpu:bool) -> dict:
    logger.info("Evaluating medical writing...")
     
    chat_handler = Gemma3ChatHandler(
        clip_model_path="./models/MedGemma/mmproj-F16.gguf",
        verbose=False
    )
    llm = Llama(
        model_path="./models/MedGemma/medgemma-4b-it-Q4_K_M.gguf",
        chat_handler=chat_handler,
        n_ctx=4096 * 14,
        use_mmap=True,
        n_threads=_CPU_CONF["threads"],
        n_gpu_layers=_GPU_CONF['gpu_layers'] if gpu else 0,
        flash_attn=_GPU_CONF["flash_attention"] if gpu else False,
        verbose=False
    )
    
    realfw = image_to_base64_data_uri(path_to_image)
    fakefw = image_to_base64_data_uri(path_to_fw_image)

    start = time.time()

    if candidate is None:
        candidate = "Candidate didn't give a respond. Set all score to 0."

    critic_to_candidate = HuaTuoExpert(candidate, True, gpu)

    output = llm.create_chat_completion(
        messages=[
                    {
                    "role": "system",
                    "content": system_rule
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": fakefw}},
                            {"type": "text", "text": user_shot_one
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_shot_one
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": realfw}},
                            {"type": "text", "text": f"""
                                REFERENCE (expert description):
                                {construct_expert_sequence(reference)}

                                CANDIDATE (to evaluate):
                                Candidate: "{candidate}"

                                CRITIC (critic to candidate statement):
                                Critic: "{critic_to_candidate['answer']}"
                                Reason for Critic: "{critic_to_candidate['reasoning']}"

                                Evaluate the Candidate's Technical Accuracy, Writing Style, 
                                Clinical Reasoning & Differential Diagnosis, Safety & Urgency 
                                Assessment, Completeness of Assessment, Diagnostic Specificity, 
                                Educational Value, Accuracy of Limitation, Structured Presentation, 
                                Clinical Actionability and give the Reason.
                            """
                            }
                        ]
                    }
            ],
            max_tokens=4096,
            temperature=0.0,
            top_p=0.9,
            stop=["```", "[STOP]"]
    )
    
    result = output['choices'][0]['message']['content']
    logger.info(f"Evaluation completed in {time.time() - start:.2f}s")
    logger.info(result)
    scores = _parse_evaluation(result, candidate, reference, factor)

    return scores

def _posthoc_failure_gate(overall: float, reason: str):

    SEVERITY_TERMS = [
        "completely inappropriate",
        "completely inaccurate",
        "completely inadequat",
        "lacks any",
        "unacceptable",
        "dangerous",
        "critical failure",
        "life-threatening",
        "potentially fatal",
        "contraindicated",
        "medically unsafe",
        "serious adverse outcome",
        "patient safety violation",
        "clinically harmful",
        "medically negligent",
        "diagnostically incorrect",
        "therapeutically wrong",
        "clinically unsound",
        "medically inaccurate",
        "below standard of care",
        "violates medical guidelines",
        "deviates from protocol",
        "fails clinical criteria",
        "requires immediate correction",
        "severe clinical error",
        "major medical mistake",
        "critical diagnostic failure",
        "emergency-level concern",
        "No reasoning provided"
    ]

    if any(term in reason.lower() for term in SEVERITY_TERMS):
        logger.warning("Failure gate triggered due to severity. This message is safe to prevent catastrophic failure")
        overall = 0.00
    else:
        overall = overall

    return overall


def _parse_evaluation(text: str, candidate:str, expert:str, factor: dict) -> dict:

    technical           = re.search(r'Technical Accuracy:\s*([0-9.]+)', text)
    writing             = re.search(r'Writing Style:\s*([0-9.]+)', text)
    clinical_reasoning  = re.search(r'Clinical Reasoning & Differential Diagnosis:\s*([0-9.]+)', text)
    safety              = re.search(r'Safety & Urgency Assessment:\s*([0-9.]+)', text)
    completeness        = re.search(r'Completeness of Assessment:\s*([0-9.]+)', text)
    specificity         = re.search(r'Diagnostic Specificity:\s*([0-9.]+)', text)
    educational         = re.search(r'Educational Value:\s*([0-9.]+)', text)
    limitations         = re.search(r'Accuracy of Limitations:\s*([0-9.]+)', text)
    structure           = re.search(r'Structured Presentation:\s*([0-9.]+)', text)
    actionability       = re.search(r'Clinical Actionability:\s*([0-9.]+)', text)
    reasoning           = re.search(r'Reasoning:\s*(.+?)(?:\n\n|\Z)', text, re.DOTALL)
    
    technical_score             = float(technical.group(1)) if technical else 0.0
    writing_score               = float(writing.group(1)) if writing else 0.0
    clinical_reasoning_score    = float(clinical_reasoning.group(1)) if clinical_reasoning else 0.0
    safety_score                = float(safety.group(1)) if safety else 0.0
    completeness_score          = float(completeness.group(1)) if completeness else 0.0
    specificity_score           = float(specificity.group(1)) if specificity else 0.0
    educational_score           = float(educational.group(1)) if educational else 0.0
    limitations_score           = float(limitations.group(1)) if limitations else 0.0
    structure_score             = float(structure.group(1)) if structure else 0.0
    actionability_score         = float(actionability.group(1)) if actionability else 0.0
    reasoning_text              = reasoning.group(1).strip() if reasoning else "No reasoning provided"
     
    if factor is None:
        logger.warning("Factor is none, defaulting to standard factoring.")
        overall = (
            technical_score             * 0.25 +           
            writing_score               * 0.10 +              
            clinical_reasoning_score    * 0.20 +    
            safety_score                * 0.15 +              
            completeness_score          * 0.10 +         
            specificity_score           * 0.05 +     
            educational_score           * 0.05 +          
            limitations_score           * 0.05 +       
            structure_score             * 0.03 +       
            actionability_score         * 0.02        
        )
    else:
        overall = (
            technical_score             * factor["TECH_S"]  +      
            writing_score               * factor["WRT_S"]   +      
            clinical_reasoning_score    * factor["CLIN_RS"] +     
            safety_score                * factor["SFTY_S"]  +    
            completeness_score          * factor["CPLT_S"]  +     
            specificity_score           * factor["SPEC_S"]  +     
            educational_score           * factor["EDU_S"]   +      
            limitations_score           * factor["LIM_S"]   +       
            structure_score             * factor["STRC_S"]  +      
            actionability_score         * factor["ACT_S"]         
        )
    
    overall = _posthoc_failure_gate(overall, reasoning_text)

    return {
        'score': overall,
        'technical_accuracy': technical_score,
        'writing_style': writing_score,
        'clinical_reasoning': clinical_reasoning_score,
        'safety_urgency': safety_score,
        'completeness': completeness_score,
        'diagnostic_specificity': specificity_score,
        'educational_value': educational_score,
        'accuracy_limitations': limitations_score,
        'structured_presentation': structure_score,
        'clinical_actionability': actionability_score,
        'reasoning': reasoning_text,
        'raw_output': text
    }


def Judge(candidate_text: str, image_path: str, fake_image_path:str, factor:dict, verbose:bool, gpu:bool) -> dict:
    start = time.time()
    ref1 = MedGemmaExpert(image_path, gpu)
    ref2 = HuaTuoExpert(ref1, False, gpu)

    references = [
        ref1,
        ref2['answer']
    ]

    if len(references) < 2:
        logger.warning(f"Found {len(references)} but required more than 1.")
        references[1] = "I'm unable to evaluate Expert 1, proceed evaluating the candidate."

    evaluation = MedGemmaScorer(references, candidate_text, image_path, fake_image_path, factor, gpu)
    end = time.time() - start

    logger.info(f"Final Score: {evaluation['score']:.3f}")
    logger.info(f"Technical: {evaluation['technical_accuracy']:.3f}, Writing: {evaluation['writing_style']:.3f}")


    if verbose:
        result = {**evaluation}
        logger.info("EVALUATION RESULTS")
        logger.info(f"Overall Score: {result['score']:.3f}")
        logger.info("DIMENSION SCORES:")
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
        logger.info("DETAILED REASONING:")
        logger.info(f"\n{result['reasoning']}\n")
        logger.info(f"Took {end:.2f} seconds to complete.")

    return {
        **evaluation
    }


if __name__ == "__main__":

    # ref = MedGemmaExpert("./examples/testing/basal.jpg", gpu=False)
    # output = HuaTuoExpert(ref, gpu=False)

    # logger.info(str(output['reasoning']))
    # logger.info(str(output['answer']))

    # Test case
    image_path = "./examples/testing/dermatitis.jpg"

    candidates = [
        '''It might be an acne due to the spots.'''   
    ]

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

    for cands in candidates:
        logger.info(f"To be evaluated: {candidates}")
        result = Judge(cands, image_path, fake_image, factor=factor, verbose=True, gpu=False)



