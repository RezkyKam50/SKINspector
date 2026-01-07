from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from loguru import logger
import re
import base64
import time

def image_to_base64_data_uri(file_path: str, mime_type: str = "image/jpeg") -> str:
    """Convert image to base64 data URI."""
    logger.info("Encoding image...")
    start = time.time()
    with open(file_path, "rb") as img_file:
        base64_data = base64.encodebytes(img_file.read()).decode("ascii")
    logger.info(f"Encoding completed in {time.time() - start:.6f}s")
    return f"data:{mime_type};base64,{base64_data}"


def MedGemmaExpert(path_to_image: str) -> str:
    logger.info("Generating reference description...")
    image_uri = image_to_base64_data_uri(path_to_image)
    chat_handler = Llava15ChatHandler(
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
        temperature=0.7,
        top_p=0.9
    )
    
    logger.info(f"Reference generated in {time.time() - start:.2f}s")
    return output['choices'][0]['message']['content']


def MedGemmaScorer(reference: str, candidate: str, path_to_image: str) -> dict:
    """
    Evaluate candidate description against reference for medical writing quality.
    
    Returns dict with:
        - score: float 0.0-1.0
        - technical_accuracy: float 0.0-1.0
        - writing_style: float 0.0-1.0
        - reasoning: str
    """
    logger.info("Evaluating medical writing...")
    
    image_uri = image_to_base64_data_uri(path_to_image)
    chat_handler = Llava15ChatHandler(
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
                "content": """
                
                You are an unforgiving dermatology expert evaluating medical writing quality. 
                Compare the Candidate description to the Reference and image.
                You have to punish the scoring if the Candidate didn't give the correct terminology, medical procedure and reasoning for the image.
                
                Score two dimensions (0.0-1.0 each):

                TECHNICAL ACCURACY (matches image and clinical reality):
                - 1.0: Perfect - Flawless primary diagnosis AND comprehensive morphological analysis (shape, size, color, borders, texture, distribution) AND complete relevant differential diagnoses AND precise anatomical location with regional context
                - 0.9: Near Perfect - Correct diagnosis with all key features accurately described, complete differentials, may have one extremely minor omission (e.g., secondary textural detail)
                - 0.8: Excellent - Correct diagnosis, all major features present, minor gaps in secondary characteristics or differential ranking
                - 0.7: Very Good - Correct diagnosis but missing 1-2 important morphological descriptors OR differentials incomplete
                - 0.6: Good - Correct diagnosis but limited feature description (missing 2-3 key elements) OR differentials present but not well-justified
                - 0.5: Acceptable - Correct general diagnostic category but non-specific (e.g., "dermatitis" without subtype), basic features identified but incomplete
                - 0.4: Borderline - Diagnosis questionable or overly broad, several features misidentified or omitted, weak differential list
                - 0.3: Poor - Wrong diagnosis within plausible category OR correct diagnosis with majority of features incorrect/missing
                - 0.2: Very Poor - Incorrect diagnostic category (e.g., inflammatory when neoplastic) OR fundamental misidentification of visible characteristics
                - 0.1: Critical Failure - Dangerous misdiagnosis with serious clinical implications (e.g., dismissing malignancy, missing urgent conditions)
                - 0.0: Unacceptable - No medically relevant analysis OR fabricated findings not present in image OR refusal to evaluate medical content

                WRITING STYLE (medical professionalism):
                - 1.0: Exemplary - Consistently precise dermatological terminology, systematic structured format (primary morphology→distribution→configuration→location→diagnosis→differentials), appropriate conciseness, professional tone, clinically actionable
                - 0.9: Excellent - Precise medical terminology throughout, clear logical structure, professional presentation with minimal stylistic improvements possible
                - 0.8: Very Good - Strong medical terminology with rare (1-2) less precise terms, well-organized, appropriate detail level
                - 0.7: Good - Predominantly medical language but 2-3 imprecise descriptors, generally structured but could be more systematic
                - 0.6: Acceptable - Adequate medical terminology mixed with vague terms (e.g., "concerning appearance"), basic organization evident
                - 0.5: Borderline - Inconsistent terminology, poorly organized, either excessively verbose or insufficiently detailed
                - 0.4: Poor - Frequent lay terminology (e.g., "bump," "spot," "mark"), unclear structure, missing standard descriptive categories
                - 0.3: Very Poor - Predominantly colloquial language, disorganized or rambling presentation, inappropriate brevity or verbosity
                - 0.2: Severely Inadequate - Minimal medical terminology, incoherent organization, reads as casual observation rather than clinical assessment
                - 0.1: Unprofessional - Casual or emotional language, lacks professional structure, contains irrelevant commentary
                - 0.0: Unacceptable - Non-medical language throughout OR inappropriate content (humor, personal opinions, off-topic information) OR incomprehensible presentation

                OUTPUT FORMAT:
                Technical Accuracy: [score]
                Writing Style: [score]
                Reasoning: [single paragraph of explanation]
                
                """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": f"""
                            REFERENCE (expert description):
                            "{reference}".

                            CANDIDATE (to evaluate):
                            "{candidate}".

                            Evaluate the Candidate's technical accuracy and writing style."""
                    }
                ]
            }
        ],
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        stop=["USER:", "ASSISTANT:", "Assistant", "\n\nUSER", "\n\nASSISTANT", "```", "\nFinal Answer:"]
    )
    
    result_text = output['choices'][0]['message']['content']
    logger.info(f"Evaluation completed in {time.time() - start:.2f}s")
    scores = _parse_evaluation(result_text)

    return scores


def _parse_evaluation(text: str) -> dict:
    technical = re.search(r'Technical Accuracy:\s*([0-9.]+)', text)
    writing = re.search(r'Writing Style:\s*([0-9.]+)', text)
    reasoning = re.search(r'Reasoning:\s*(.+?)(?:\n\n|\Z)', text, re.DOTALL)
    
    technical_score = float(technical.group(1)) if technical else 0.0
    writing_score = float(writing.group(1)) if writing else 0.0
    reasoning_text = reasoning.group(1).strip() if reasoning else "No reasoning provided"
     
    overall = (technical_score * 0.6) + (writing_score * 0.4)
    
    return {
        'score': overall,
        'technical_accuracy': technical_score,
        'writing_style': writing_score,
        'reasoning': reasoning_text,
        'raw_output': text
    }


def evaluate_description(candidate_text: str, image_path: str) -> dict:
    """
    Main evaluation function: evaluate a candidate description.
    
    Args:
        candidate_text: Medical description to evaluate
        image_path: Path to dermatology image
    
    Returns:
        dict with scores and feedback
    """
    # Generate reference
    reference = MedGemmaExpert(image_path)
    logger.info(f"Reference: {reference[:200]}...")
    evaluation = MedGemmaScorer(reference, candidate_text, image_path)
    
    logger.info(f"Final Score: {evaluation['score']:.3f}")
    logger.info(f"Technical: {evaluation['technical_accuracy']:.3f}, Writing: {evaluation['writing_style']:.3f}")
    
    return {
        'reference': reference,
        **evaluation
    }


if __name__ == "__main__":
    # Test case
    candidate_description = '''
    This image contains Basal Cell Carcinoma 
    '''
    
    image_path = "./examples/testing/basal_1.jpg"
    
    result = evaluate_description(candidate_description, image_path)
    
    overall_score = result['score']
    technical_score = result['technical_accuracy']
    writing_score = result['writing_style']
    reason_for_score = result['reasoning']

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Score: {result['score']:.3f}")
    print(f"Technical Accuracy: {result['technical_accuracy']:.3f}")
    print(f"Writing Style: {result['writing_style']:.3f}")
    print(f"\nReasoning: {result['reasoning']}")
    print("="*60)