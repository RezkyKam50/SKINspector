rl_system_message = '''
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant 
    first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
    <think> reasoning process here </think><answer> answer here </answer>"
'''

query_message="""
Perform a comprehensive dermatological analysis of the provided image. Structure your response using the following sections:
A. Visual Description: Objectively describe the clinical findings. Include:
    Primary Morphology: (e.g., macule, papule, plaque, vesicle, pustule, nodule).
    Color & Shape: (e.g., erythematous, violaceous, annular, nummular, serpiginous).
    Scale/Crust/Erosion: Note the presence and characteristics of any surface changes.
    Distribution & Pattern: (e.g., localized/generalized, symmetrical, follicular, linear).
B. Most Likely Diagnosis: Based on the visual description, what is the single most probable condition? Provide a brief rationale linking the description to the diagnosis.
C. Medical Treatment Options: List 2-3 first-line, evidence-based treatment modalities for the most likely diagnosis. Specify the type (e.g., Topical, Systemic, Procedural).
"""