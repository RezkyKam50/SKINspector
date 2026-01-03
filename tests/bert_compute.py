from ..utils.qwen_utils.bert_embedding import compute_prf1, safe_divide 

from loguru import logger
import time
 
if __name__ == "__main__":
    predictions = [
        "Administer aspirin for acute myocardial infarction management",
        "Patient shows signs of hypertension requiring immediate treatment",
        "Prescribe insulin therapy for type 2 diabetes control",
        "Recommend physical therapy for lower back pain relief",
        "Start antibiotic treatment for bacterial pneumonia infection",
        "Administer epinephrine for severe allergic reaction symptoms",
        "Initiate oxygen therapy for chronic obstructive pulmonary disease",
        "Prescribe antidepressants for major depressive disorder treatment",
        "Recommend surgical intervention for acute appendicitis case",
        "Start chemotherapy regimen for stage 3 breast cancer",
        "Administer corticosteroids for severe asthma exacerbation",
        "Prescribe anticoagulation therapy for deep vein thrombosis",
        "Initiate dialysis treatment for end-stage renal disease",
        "Recommend lifestyle modifications for metabolic syndrome management",
        "Start antiviral medication for influenza infection treatment",
        "Administer beta blockers for heart failure management",
        "Prescribe proton pump inhibitors for gastroesophageal reflux disease"
    ]

    references = [
        "Avoid aspirin administration in hemorrhagic stroke patients",
        "Monitor blood pressure and adjust medication accordingly",
        "Adjust oral hypoglycemic agents and monitor glucose levels regularly",
        "Consider epidural steroid injections if conservative management fails",
        "Obtain sputum culture and chest X-ray before initiating treatment",
        "Monitor vital signs and provide supportive care in emergency",
        "Assess arterial blood gases and adjust flow rate appropriately",
        "Evaluate patient response and adjust dosage after 4-6 weeks",
        "Perform CT scan to confirm diagnosis before surgical planning",
        "Discuss treatment options and potential side effects with patient",
        "Monitor peak flow measurements and adjust medication doses",
        "Check INR levels regularly and adjust warfarin dosage",
        "Monitor electrolytes and fluid balance during treatment sessions",
        "Counsel on diet, exercise, and weight management strategies",
        "Start treatment within 48 hours of symptom onset",
        "Titrate medication slowly while monitoring cardiac function",
        "Advise taking medication 30 minutes before meals for effectiveness"
    ]

    start = time.time()
    P, R, F1 = compute_prf1(
        cands=predictions,
        refs=references,
        model_path="./models/pubmedbert-base-embeddings",
        max_length=512,
        device="cpu",
        fp16=False,
        batch_size=1
    )
    end = time.time() - start
    logger.info(f"Non Batched ver took {end} seconds")
    
    precision_avg           = safe_divide(P.sum().item(), len(P))
    recall_avg              = safe_divide(R.sum().item(), len(R))
    f1_avg                  = safe_divide(F1.sum().item(), len(F1))

    print(f"Precision: {precision_avg}")
    print(f"Recall: {recall_avg}")
    print(f"F1: {f1_avg}")
     
    start = time.time()
    P, R, F1 = compute_prf1(
        cands=predictions,
        refs=references,
        model_path="./models/pubmedbert-base-embeddings",
        max_length=512,
        batch_size=15,
        device="cpu",
        fp16=False,
    )
    end = time.time() - start
    logger.info(f"Batched ver took {end} seconds")
    
    precision_avg           = safe_divide(P.sum().item(), len(P))
    recall_avg              = safe_divide(R.sum().item(), len(R))
    f1_avg                  = safe_divide(F1.sum().item(), len(F1))

    print(f"Precision: {precision_avg}")
    print(f"Recall: {recall_avg}")
    print(f"F1: {f1_avg}")
