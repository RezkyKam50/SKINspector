
def construct_expert_sequence(reference):
    exp_chain = []
    for idx, exp in enumerate(reference):
        exp_chain.append(f'Expert {idx + 1}: "{exp}"')
    exp_chain = "\n".join(exp_chain)
    
    return exp_chain

system_rule = """        
SCORING RULES (MANDATORY):
- If the Candidate diagnosis is incorrect OR absent = Technical Accuracy MUST be less than 0.2
- If no differential diagnosis is provided = Clinical Reasoning MUST be less than 0.2
- If no safety or urgency guidance is provided for a potentially malignant lesion = Safety & Urgency MUST be less than 0.2
- If the assessment is fundamentally flawed = At least 3 categories MUST be less than 0.1
- If the Candidate is clinically unacceptable = Overall performance MUST reflect near-zero scores (less than 0.1 in most categories)
- You are allowed to set all scores to 0
Do not soften scores. Use 0.0 - 0.1 when criteria are unmet.

TECHNICAL ACCURACY :
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

WRITING STYLE :
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

CLINICAL REASONING & DIFFERENTIAL DIAGNOSIS :
- 1.0: Exemplary - Comprehensive differential (3-5 appropriate alternatives), each with specific justifying features, ranked by likelihood with clear reasoning, acknowledges key distinguishing factors
- 0.9: Excellent - Strong differential list (3-4 alternatives) with good justification, logical ranking, explains primary diagnosis selection
- 0.8: Very Good - Solid differentials (2-3 alternatives) with adequate reasoning, appropriate likelihood ranking
- 0.7: Good - Reasonable differentials (2-3) but limited justification OR good justification but incomplete list
- 0.6: Acceptable - Basic differentials present (1-2) with minimal reasoning, ranking may be unclear
- 0.5: Borderline - Differentials listed but poorly justified OR only 1 alternative considered
- 0.4: Poor - Weak/implausible differentials OR no justification provided
- 0.3: Very Poor - Inappropriate differentials that don't fit clinical picture
- 0.2: Severely Inadequate - Differentials mentioned but completely incorrect
- 0.1: Critical Failure - No differential diagnosis attempted OR dangerous omissions (e.g., missing malignancy in differential)
- 0.0: Unacceptable - Refuses to provide differential OR completely fabricated alternatives

SAFETY & URGENCY ASSESSMENT :
- 1.0: Perfect - Accurate identification of all red flags, appropriate urgency level (emergency/urgent/routine), specific timeframe for evaluation, clear malignancy risk stratification
- 0.9: Excellent - Correct urgency assessment, identifies critical features, appropriate follow-up recommendations, minor details could be enhanced
- 0.8: Very Good - Appropriate urgency level, recognizes key warning signs, reasonable follow-up timeline
- 0.7: Good - Generally appropriate urgency but may lack specificity in timing OR misses one minor red flag
- 0.6: Acceptable - Basic safety assessment present, appropriate general urgency level, lacks detail
- 0.5: Borderline - Vague urgency assessment ("should be evaluated") without specific timeframe
- 0.4: Poor - Urgency level questionable OR misses important warning signs
- 0.3: Very Poor - Incorrect urgency assessment (e.g., routine for urgent condition)
- 0.2: Severely Inadequate - Dangerous under-triaging OR multiple missed red flags
- 0.1: Critical Failure - Dismisses serious/malignant condition as benign OR inappropriately reassuring
- 0.0: Unacceptable - No safety assessment OR encourages dangerous delay in concerning cases

COMPLETENESS OF ASSESSMENT :
- 1.0: Comprehensive - Systematic evaluation covering: primary morphology (macule/papule/plaque/etc.), secondary features (scale/crust/erosion), size measurements, color description, border characteristics, texture, distribution pattern, configuration, anatomical location with regional context, background skin
- 0.9: Near Complete - All major categories assessed (8-9 of above elements), minor secondary detail omitted
- 0.8: Very Complete - Strong assessment covering 7-8 key elements, organized presentation
- 0.7: Good - Covers 6-7 elements including all critical primary features
- 0.6: Acceptable - Basic assessment with 5-6 elements, may miss some secondary features
- 0.5: Borderline - Limited assessment (4-5 elements), gaps in important descriptors
- 0.4: Incomplete - Only 3-4 elements assessed, missing multiple key categories
- 0.3: Poor - Minimal assessment (2-3 elements), significant gaps in description
- 0.2: Very Incomplete - Only 1-2 basic observations, lacks systematic approach
- 0.1: Critically Incomplete - Barely mentions visible features
- 0.0: Unacceptable - No systematic assessment OR only addresses diagnosis without descriptive analysis

DIAGNOSTIC SPECIFICITY :
- 1.0: Optimally Specific - Provides most specific diagnosis possible from image (including subtypes when determinable), clearly distinguishes between clinical impression vs. histologically-confirmed diagnosis, acknowledges exact limitations
- 0.9: Highly Specific - Specific diagnosis with appropriate subtyping, minor clarifications possible
- 0.8: Very Good Specificity - Specific diagnosis provided, appropriate level of certainty expressed
- 0.7: Good Specificity - Reasonably specific but could be more precise (e.g., "melanocytic nevus" vs. "compound nevus")
- 0.6: Acceptable Specificity - Adequate specificity but somewhat general when more precision possible
- 0.5: Borderline - Overly broad diagnosis when more specificity achievable (e.g., "benign growth" instead of "seborrheic keratosis")
- 0.4: Poor Specificity - Vague diagnosis (e.g., "skin lesion," "rash") when specific diagnosis evident
- 0.3: Very Vague - Non-specific diagnosis category only (e.g., "dermatitis" without further classification)
- 0.2: Severely Vague - Extremely broad terminology that provides minimal clinical value
- 0.1: Unhelpfully Vague - Diagnosis so general it's nearly meaningless (e.g., "skin condition")
- 0.0: Unacceptable - No diagnosis provided OR inappropriately certain when uncertainty warranted OR claims definitive diagnosis when impossible from image alone

EDUCATIONAL VALUE :
- 1.0: Exemplary Teaching - Clear explanation of diagnostic reasoning process, highlights key distinguishing features with educational emphasis, provides relevant clinical pearls, uses teachable moments, balances technical accuracy with accessibility
- 0.9: Excellent Educational Content - Strong explanatory framework, identifies learning points, helpful context provided
- 0.8: Very Good Educational Value - Good reasoning explanation, highlights important features, some teaching elements
- 0.7: Good Educational Approach - Explains thinking process, identifies key features, somewhat instructive
- 0.6: Acceptable Educational Content - Basic reasoning shared, some explanatory value
- 0.5: Borderline Educational - Minimal explanation of reasoning, mostly descriptive without teaching
- 0.4: Limited Educational Value - Little explanation of "why," difficult to learn from
- 0.3: Poor Educational Quality - No reasoning explained, purely diagnostic statement
- 0.2: Minimal Educational Content - No teaching value, unexplained conclusions
- 0.1: Non-Educational - Provides answer without any instructive content
- 0.0: Unacceptable - No educational framework OR actively confusing/misleading explanations

ACCURACY OF LIMITATIONS :
- 1.0: Perfect Calibration - Precisely identifies what can vs. cannot be determined from image, appropriate diagnostic certainty level, clear about when biopsy/histopathology needed, honest about image quality constraints, acknowledges clinical history gaps
- 0.9: Excellent Calibration - Well-calibrated certainty, acknowledges key limitations, appropriate caveats
- 0.8: Very Good Calibration - Generally appropriate certainty, mentions important limitations
- 0.7: Good Calibration - Reasonable certainty level, acknowledges some limitations
- 0.6: Acceptable Calibration - Basic acknowledgment of limitations, mostly appropriate confidence
- 0.5: Borderline Calibration - Vague about limitations OR slightly overconfident/underconfident
- 0.4: Poor Calibration - Inappropriate certainty level (too confident or too hesitant)
- 0.3: Very Poor Calibration - Significant overconfidence OR excessive hedging when diagnosis clear
- 0.2: Severely Miscalibrated - Claims certainty when impossible OR refuses diagnosis when reasonable
- 0.1: Dangerously Miscalibrated - Definitive diagnosis claimed without appropriate caveats for serious conditions
- 0.0: Unacceptable - No acknowledgment of limitations when critical OR completely inappropriate confidence level

STRUCTURED PRESENTATION :
- 1.0: Perfectly Organized - Follows systematic clinical format (e.g., Description→Location→Morphology→Differential→Recommendation), logical flow, appropriate use of formatting, easy to parse, professional structure
- 0.9: Excellently Structured - Clear systematic organization, logical progression, well-formatted
- 0.8: Very Well Structured - Good organization with clear sections, easy to follow
- 0.7: Well Structured - Organized presentation, logical flow, minor structural improvements possible
- 0.6: Acceptably Structured - Basic organization present, generally followable
- 0.5: Borderline Structure - Some organization but inconsistent or unclear progression
- 0.4: Poorly Structured - Disorganized presentation, difficult to follow, jumps between topics
- 0.3: Very Poorly Structured - Minimal organization, confusing flow, lacks clear sections
- 0.2: Severely Disorganized - Chaotic presentation, no clear structure, hard to extract information
- 0.1: Incoherent Structure - Nearly impossible to follow, completely disorganized
- 0.0: Unacceptable - No discernible structure OR formatting that obscures content

CLINICAL ACTIONABILITY :
- 1.0: Fully Actionable - Provides clear, specific next steps (biopsy, specialist referral, timeframe, monitoring plan), practical management guidance appropriate to diagnosis and urgency, includes what to watch for
- 0.9: Highly Actionable - Clear recommendations with specific guidance, minor details could be added
- 0.8: Very Actionable - Good specific recommendations, appropriate management direction
- 0.7: Actionable - Reasonable next steps provided, somewhat specific guidance
- 0.6: Acceptably Actionable - Basic recommendations present, lacks some specificity
- 0.5: Borderline Actionable - Vague recommendations ("see doctor") without specific guidance
- 0.4: Limited Actionability - Minimal guidance on next steps, lacks practical advice
- 0.3: Poor Actionability - Very vague or generic recommendations
- 0.2: Minimal Actionability - Almost no practical guidance provided
- 0.1: Non-Actionable - No recommendations OR inappropriate advice
- 0.0: Unacceptable - Dangerous recommendations OR actively discourages appropriate care


OUTPUT FORMAT:
Technical Accuracy: [score]
Writing Style: [score]
Clinical Reasoning & Differential Diagnosis: [score]
Safety & Urgency Assessment: [score]
Completeness of Assessment: [score]
Diagnostic Specificity: [score]
Educational Value: [score]
Accuracy of Limitations: [score]
Structured Presentation: [score]
Clinical Actionability: [score]
Reasoning: [reasoning] [STOP]

"""

user_shot_one ="""
REFERENCE (expert description):
Expert 1: "Okay, here's a clinical description of the skin lesion based on the provided image, using the requested terminology:

**Clinical Description:**

The image shows a raised, erythematous lesion with a central umbilication and crusting. The lesion is located on the face, specifically on the cheek.

**1. Morphology:**

*   **Lesion Type:** Ulcerated nodule
*   **Size:** Approximately 1.5 cm in diameter.
*   **Shape:** Irregular, with a central umbilication.
*   **Color:** Erythematous (red) with areas of crusting.
*   **Borders:** Ill-defined, with surrounding erythema.
*   **Texture:** Ulcerated, crusting, and potentially scabrous.

**2. Location and Distribution:**

*   **Location:** Cheek, face.
*   **Distribution:** Single lesion.

**3. Key Clinical Features:**

*   Erythematous base.
*   Central umbilication.
*   Crusting.
*   Ulceration.

**4. Most Likely Diagnosis with 2-3 Differentials:**

*   **Most Likely:** Basal Cell Carcinoma (BCC)
*   **Differentials:**
    *   Squamous Cell Carcinoma (SCC)
    *   Malignant Melanoma

**5. Risk Factors (ABCDE criteria if pigmented):**

*   **Sun Exposure:** Prolonged and unprotected sun exposure is a major risk factor for all three diagnoses.
*   **Age:** Risk increases with age.
*   **Fair Skin:** Individuals with fair skin are more susceptible.
*   **History of Sunburns:** A history of sunburns increases risk.
*   **Immunosuppression:** Immunosuppressed individuals are at higher risk.
*   **Previous Radiation Therapy:** Prior radiation exposure increases risk.
*   **Genetic Predisposition:** Family history of skin cancer.

**Important Considerations:**

*   This is a preliminary assessment based on a single image. A definitive diagnosis requires a clinical examination, potentially a biopsy, and histopathological analysis.
*   The ABCDE criteria are primarily relevant for pigmented lesions (melanomas). While they can be applied to other lesions, they are not as crucial for non-pigmented lesions like BCC.
*   The presence of ulceration is a significant feature that points towards a diagnosis of BCC or SCC.

**Disclaimer:** This information is for educational purposes only and should not be considered medical advice. Always consult with a qualified dermatologist for diagnosis and treatment of skin conditions. 
Expert 2: "The clinical description provided is thorough and well-structured, effectively covering the morphology, location, and key clinical features of the skin lesion. The assessment is based on a raised, erythematous lesion with a central umbilication and crusting, located on the cheek, which aligns with the typical presentation of basal cell carcinoma (BCC). The use of the ABCDE criteria is relevant, although primarily designed for pigmented lesions, they can still be applied to non-pigmented lesions like BCC to assess features such as size and evolution.
The differential diagnoses of squamous cell carcinoma (SCC) and malignant melanoma are also appropriate, as they are common skin cancers that can present similarly. However, it would be beneficial to consider additional differentials, such as pyogenic granuloma or benign keratosis, which could have overlapping features.
The risk factors discussed, including sun exposure, age, fair skin, history of sunburns, and immunosuppression, are all relevant and important considerations. The mention of previous radiation therapy and genetic predisposition is also pertinent, as these can significantly impact the risk of developing skin cancer.
The assessment is well-supported by the described features, and the conclusion that BCC is the most likely diagnosis is reasonable. However, it is crucial to emphasize that a definitive diagnosis requires a clinical examination and potentially a biopsy, along with histopathological analysis.
In summary, the clinical description is comprehensive, and the diagnostic reasoning is well-supported. However, it would be beneficial to consider additional differentials and risk factors to ensure a thorough evaluation. A biopsy is essential to confirm the diagnosis."

CRITIC (critic to candidate statement):
Critic: "The dermatological assessment has several weaknesses and areas for improvement. 
First, there is a lack of critical clinical details, such as lesion size and duration, which are essential for a thorough evaluation. The absence of these details creates significant uncertainty in the diagnostic process. 
Second, the primary diagnosis of basal cell carcinoma is not adequately justified, given the absence of vesicles or pustules, which are not typical for this condition. The unusual yellowish-white appearance further questions the diagnosis and suggests diagnostic uncertainty. 
Third, alternative diagnoses are not clearly considered. The assessment mentions pathergy and contact dermatitis, but these seem unlikely without documented trauma and an isolated papule pattern. There should be a more comprehensive exploration of potential differential diagnoses, such as viral infections or other skin conditions. 
Fourth, the risk factors and warning signs are inadequately addressed. Without a thorough history, including the patient's immunization status and past medical conditions, it's challenging to provide informed and appropriate treatment decisions. 
Fifth, the proposed diagnostic testing—viral culture or PCR—seems unconventional for confirming basal cell carcinoma. This approach may indicate an uncertainty about the diagnosis and a leaning towards a viral etiology, which is not typical for basal cell carcinoma. 
Finally, the clinical reasoning has gaps, particularly in considering the absence of imaging confirmation. The assessment relies heavily on appearance and description, which is limited. A more systematic and comprehensive evaluation, incorporating clinical, laboratory, and imaging findings, is necessary to accurately diagnose and manage this skin condition. 
Overall, the assessment would benefit from additional detail on clinical presentation, a broader consideration of differential diagnoses, and a more conventional diagnostic approach."
Reason for Critic: "Let's see, the assessment here is trying to figure out what's going on with this skin lesion. They mention basal cell carcinoma, but there's something off. The description doesn't have vesicles or pustules, which is kind of unusual for basal cell carcinoma. So, there's already some uncertainty there. 
They say they're considering other things like pathergy and contact dermatitis, but those don't really fit either. No trauma is mentioned, and it's just a single papule. So, those are a bit of a stretch. 
Next, they talk about needing more information. A thorough history is important, and they mention checking the timeline of symptoms, but they don't dive into what else they need from the patient. 
Oh, and then they suggest using viral culture or PCR to confirm the diagnosis. That seems a bit out of the ordinary. It's not common for basal cell carcinoma to be confirmed that way, so this seems a bit off. 
Then there's the mention of considering alternative diagnoses. That's good, but they don't specify what exactly those are. They just say to look at the patient's medical history. 
Let's think about what's missing here. The description is based on appearances, and they're trying to make a diagnosis without any imaging. That's pretty limited. Normally, you'd expect more detailed information about the lesion, like its size or how long it's been there. And a complete medical history would give a much clearer picture. 
There's also a lack of alternative diagnoses being clearly considered. Sure, they mention looking at the patient's past medical conditions, but they don't specify which conditions or how they relate to the lesion. 
Then there's the whole diagnostic testing part. Viral culture or PCR isn't really a standard for basal cell carcinoma. It seems like they might be leaning towards something viral, which isn't typical for this diagnosis. 
In conclusion, this assessment is missing a lot of details that are crucial for a proper evaluation. There's no clear plan for what they're going to do next, and the diagnostic testing they propose is unconventional. Overall, this evaluation could benefit from more concrete evidence and a more systematic approach."

CANDIDATE (to evaluate):
Candidate: "The clinical presentation raises questions about the diagnosis. 
    Basal cell carcinoma typically manifests with vesicles or pustules, 
    yet these features are absent from the description, and the yellowish-white appearance 
    of the papule doesn't align with usual presentations, indicating diagnostic uncertainty. 
    While pathergy and contact dermatitis have been included in the differential diagnosis, 
    both appear unlikely given the absence of documented trauma and the isolated central papule 
    pattern. A thorough clinical assessment and comprehensive patient history exploring symptom 
    timeline, associated features, and relevant medical background are essential next steps. 
    If basal cell carcinoma remains suspected despite the atypical presentation, diagnostic testing 
    through viral culture or PCR analysis would be valuable for confirmation. The current findings 
    suggest basal cell carcinoma as the leading possibility, but the unusual characteristics—particularly 
    the absence of vesicular lesions—warrant careful consideration of alternative diagnoses, including 
    review of the patient's immunization history and past medical conditions to guide appropriate treatment decisions."

                                Evaluate the Candidate's Technical Accuracy, Writing Style, 
                                Clinical Reasoning & Differential Diagnosis, Safety & Urgency 
                                Assessment, Completeness of Assessment, Diagnostic Specificity, 
                                Educational Value, Accuracy of Limitation, Structured Presentation, 
                                Clinical Actionability and give the Reason.
"""

assistant_shot_one=f"""
Technical Accuracy: 0.2  
Writing Style: 0.2  
Clinical Reasoning & Differential Diagnosis: 0.0  
Safety & Urgency Assessment: 0.0  
Completeness of Assessment: 0.2  
Diagnostic Specificity: 0.0  
Educational Value: 0.2  
Accuracy of Limitation: 0.0  
Structured Presentation: 0.0
Clinical Actionability: 0.0  

Reasoning: The candidate response demonstrates fundamental clinical inadequacies that align with the critic's concerns about missing critical details and unconventional diagnostic approaches. While Expert 1 provides a comprehensive assessment including morphology (ulcerated nodule, 1.5cm, irregular shape), location (cheek), and clear differential diagnoses (BCC, SCC, malignant melanoma), the candidate fails to establish any structured diagnostic framework. Expert 2 emphasizes the importance of clinical examination and biopsy for definitive diagnosis, yet the candidate focuses on atypical features ("yellowish-white appearance," "absence of vesicular lesions") without providing the thorough clinical assessment or patient history that both experts deem essential. The critic correctly identifies the unconventional approach of suggesting viral culture or PCR for suspected basal cell carcinoma, which the candidate perpetuates despite this being inappropriate for BCC diagnosis. Most critically, the candidate presents diagnostic uncertainty and mentions alternative diagnoses without actually listing them or providing risk factor analysis (sun exposure, age, fair skin) that Expert 1 identifies as crucial. The response lacks the anatomical specificity, size measurements, ABCDE criteria application, and clear clinical actionability that both experts demonstrate, resulting in an assessment that would be clinically unusable and potentially unsafe due to the absence of urgency guidance for an ulcerated facial lesion. [STOP]
"""