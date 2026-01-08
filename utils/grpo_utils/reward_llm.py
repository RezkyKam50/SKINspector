from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from rl_prompt import *
from loguru import logger
import re
import base64
import time

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


def MedGemmaExpert(path_to_image: str) -> str:
    logger.info("Generating reference description...")
    image_uri = image_to_base64_data_uri(path_to_image)
    chat_handler = Gemma3ChatHandler(
        clip_model_path="./models/MedGemma/mmproj-F16.gguf", 
        verbose=False
    )
    llm = Llama(
        model_path="./models/MedGemma/medgemma-4b-it-Q4_K_M.gguf",
        chat_format="gemma",
        n_ctx=4096 * 12,
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

                Use precise medical terminology. Be concise but comprehensive."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": "Provide a clinical description of this dermatological finding."}
                ]
            }
        ],
        max_tokens=512,
        temperature=0.4,
        top_p=0.95
    )
    
    logger.info(f"Reference generated in {time.time() - start:.2f}s")
    return output['choices'][0]['message']['content']


def MedGemmaScorer(reference: str, candidate: str, path_to_image: str, path_to_fw_image:str, factor:dict) -> dict:
    """
    Evaluate candidate description against reference for medical writing quality.
    
    Returns dict with:
        - score: float 0.0-1.0
        - technical_accuracy: float 0.0-1.0
        - writing_style: float 0.0-1.0
        - reasoning: str
    """
    logger.info("Evaluating medical writing...")
     
    chat_handler = Gemma3ChatHandler(
        clip_model_path="./models/MedGemma/mmproj-F16.gguf",
        verbose=False
    )
    llm = Llama(
        model_path="./models/MedGemma/medgemma-4b-it-Q4_K_M.gguf",
        # chat_format="gemma",
        n_ctx=4096 * 14,
        chat_handler=chat_handler,
        verbose=False
    )
    
    realfw = image_to_base64_data_uri(path_to_image)
    fakefw = image_to_base64_data_uri(path_to_fw_image)

    start = time.time()
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
                                Expert: "{reference}"
                                CANDIDATE (to evaluate):
                                Candidate: "{candidate}"

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

def _posthoc_failure_gate(overall: float, expert: str, candidate: str) -> float:

    CRITICAL_MISMATCHES = [
        (["allergic", "allergy", "allergic reaction"], ["basal cell carcinoma", "bcc", "malignancy"]),
        (["eczema", "dermatitis", "inflammatory"], ["melanoma", "malignant melanoma"]),
        (["fungal", "tinea", "candida"], ["squamous cell carcinoma", "scc"]),
        (["benign", "harmless", "non-cancerous"], ["melanoma", "aggressive malignancy"]),
        (["cosmetic", "aesthetic concern"], ["urgent", "malignant", "biopsy required"]),
        (["acne", "pimple", "comedone"], ["basal cell carcinoma", "nodular bcc"]),
        (["age spot", "sun spot", "lentigo"], ["melanoma", "lentigo maligna"]),
        (["mole", "nevus", "beauty mark"], ["amelanotic melanoma", "nodular melanoma"]),
        (["wart", "verruca", "viral lesion"], ["squamous cell carcinoma", "keratoacanthoma"]),
        (["psoriasis", "plaque", "scaling"], ["cutaneous t-cell lymphoma", "mycosis fungoides"]),
        (["rosacea", "facial redness"], ["lupus", "systemic disease"]),
        (["seborrheic keratosis", "benign growth"], ["pigmented basal cell carcinoma"]),
        (["cherry angioma", "vascular lesion"], ["kaposi sarcoma", "angiosarcoma"]),
        (["skin tag", "acrochordon", "fibroma"], ["dermatofibrosarcoma protuberans", "sarcoma"]),
        (["hives", "urticaria", "allergic"], ["urticarial vasculitis", "systemic vasculitis"]),
        (["insect bite", "bug bite"], ["basal cell carcinoma", "ulcerated lesion"]),
        (["bruise", "contusion", "trauma"], ["purpura", "thrombocytopenia", "leukemia cutis"]),
        (["dry skin", "xerosis"], ["ichthyosis", "cutaneous t-cell lymphoma"]),
        (["ingrown hair", "folliculitis"], ["squamous cell carcinoma", "perineural invasion"]),
        (["heat rash", "miliaria"], ["cutaneous lymphoma"]),
        (["freckle", "ephelis"], ["lentigo maligna melanoma"]),
        (["cyst", "epidermoid cyst"], ["metastatic carcinoma", "subcutaneous malignancy"]),
        (["keloid", "hypertrophic scar"], ["dermatofibrosarcoma protuberans"]),
        (["vitiligo", "depigmentation"], ["hypopigmented mycosis fungoides"]),
        (["lipoma", "fatty tumor"], ["liposarcoma", "soft tissue sarcoma"]),
    ]
    
    candidate_lower = candidate.lower()
    expert_lower = expert.lower()
    
    for wrong_terms, correct_terms in CRITICAL_MISMATCHES:
        matched_wrong = [term for term in wrong_terms if term in candidate_lower]
        matched_correct = [term for term in correct_terms if term in expert_lower]
        
        if matched_wrong and matched_correct:
            logger.warning(
                f"Failure gate triggered due to mismatch. "
                f"Candidate contained: {matched_wrong}, "
                f"Expert contained: {matched_correct}. "
                f"This message is safe to prevent catastrophic failure"
            )
            return 0.0
        
    return overall

def _posthoc_veto_gate(overall: str, reason: str):

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
        logger.warning("Veto gate triggered due to severity. This message is safe to prevent catastrophic failure")
        overall = min(overall, 0.05)
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
    
    overall = _posthoc_veto_gate(overall, reasoning_text)
    overall = _posthoc_failure_gate(overall, expert, candidate)

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


def Judge(candidate_text: str, image_path: str, fake_image_path:str, factor:dict) -> dict:
    """
    Main evaluation function: evaluate a candidate description.
    
    Args:
        candidate_text: Medical description to evaluate
        image_path: Path to dermatology image
        fake_image_path: Path to sample image for fewshot

    Returns:
        dict with scores and feedback
    """
    # Generate reference
    reference = MedGemmaExpert(image_path)
    logger.info(f"Reference: {reference[:200]}...")
    evaluation = MedGemmaScorer(reference, candidate_text, image_path, fake_image_path, factor)
    
    logger.info(f"Final Score: {evaluation['score']:.3f}")
    logger.info(f"Technical: {evaluation['technical_accuracy']:.3f}, Writing: {evaluation['writing_style']:.3f}")
    
    return {
        'reference': reference,
        **evaluation
    }


if __name__ == "__main__":
    # Test case
    image_path = "./examples/testing/basal.jpg"
    candidate_description='''
    what the hell
    '''

    candidates = (
        '''
        A. Visual Description:  
        - Primary Morphology: A well-demarcated, raised lesion with a distinct, irregular border. The surface appears smooth and slightly erythematous, not clearly scaled or crusted, but with a fine, vascular network visible (indicative of inflammation). The lesion is partially obscured by a dark, possibly ocular, area on the right, suggesting proximity to an eye or adjacent mucosal surface.  
        - Color & Shape: The lesion presents as erythematous to violaceous, with a slightly dusky hue and an irregular, non-uniform contour. The borders appear blurred and indistinct, blending into the surrounding skin in a way consistent with inflammatory or early proliferative processes.  
        - Scale/Crust/Erosion: No scales, crusts, or erosions are evident. The surface is smooth and homogeneous, though the presence of fine vascularization suggests active vascular changes.  
        - Distribution & Pattern: The lesion is localized, not symmetrical, and appears to occur in a dermatologic context, potentially near the ocular or mucosal area, with no other lesion visible in the frame.  

        B. Most Likely Diagnosis:  
        Most Likely Diagnosis: Inflammatory or reactive dermatosis such as
        ''',
        '''
        This image shows a close-up view of a skin disease characterized by red, inflamed patches on the skin, caused by an overproduction of skin cells, which may require treatment
        ''',
        '''
        This image shows a close-up view of a skin disease characterized by red, inflamed patches on the skin, likely caused by an allergic reaction to a specific substance. Treatment options include topical creams and avoiding the allergen 
        '''
    )

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
        result = Judge(cands, image_path, fake_image, factor=factor)
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'[]'*80}\n")
        
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


