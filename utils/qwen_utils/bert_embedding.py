from transformers import AutoTokenizer, AutoModel
from loguru import logger
import torch
import numpy as np

def safe_divide(numerator, denominator, default=0.0):
    try:
        if denominator != 0:
            logger.info(f"Denominator is {denominator}")
        return numerator / denominator if denominator != 0 else default
    
    except (TypeError, ZeroDivisionError):
        logger.error(f"Zero division for metrics, returning {default}.")
        return default

def compute_prf1(cands, refs, model_path, max_length, batch_size, device, fp16):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        if fp16:
            model.eval().half()
            logger.info(f"Model '{model.__class__.__name__}' loaded in FP16 with {model.num_parameters()} parameters")
        else:
            model.eval()
            logger.info(f"Model '{model.__class__.__name__}' loaded in FP32 with {model.num_parameters()} parameters")
    except Exception as e:
        logger.error(f"Error in loading model and tokenizer: {e}")
        return np.array([]), np.array([]), np.array([])
    
    P_scores, R_scores, F1_scores = [], [], []
    
    with torch.no_grad():
        try:
            if batch_size > 1:
                for batch_idx in range(0, len(cands), batch_size):
                    batch_cands = cands[batch_idx:batch_idx + batch_size]
                    batch_refs = refs[batch_idx:batch_idx + batch_size]
                     
                    cand_tokens = tokenizer(
                        batch_cands, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                    
                    ref_tokens = tokenizer(
                        batch_refs, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                     
                    cand_embs = model(**cand_tokens).last_hidden_state   
                    ref_embs = model(**ref_tokens).last_hidden_state
                     
                    for i in range(len(batch_cands)):
                        cand_emb = cand_embs[i]
                        ref_emb = ref_embs[i]
                        
                        cand_input_ids = cand_tokens['input_ids'][i]
                        ref_input_ids = ref_tokens['input_ids'][i]
                         
                        cand_mask = (cand_input_ids != tokenizer.cls_token_id) & \
                                    (cand_input_ids != tokenizer.sep_token_id) & \
                                    (cand_input_ids != tokenizer.pad_token_id) & \
                                    (cand_input_ids != tokenizer.mask_token_id) & \
                                    (cand_input_ids != tokenizer.unk_token_id)
                        
                        ref_mask = (ref_input_ids != tokenizer.cls_token_id) & \
                                   (ref_input_ids != tokenizer.sep_token_id) & \
                                   (ref_input_ids != tokenizer.pad_token_id) & \
                                   (ref_input_ids != tokenizer.mask_token_id) & \
                                   (ref_input_ids != tokenizer.unk_token_id)
                         
                        cand_emb = cand_emb[cand_mask]
                        ref_emb = ref_emb[ref_mask]
                        
                        if cand_emb.size(0) == 0 or ref_emb.size(0) == 0:
                            P_scores.append(0.0)
                            R_scores.append(0.0)
                            F1_scores.append(0.0)
                            continue
                         
                        cand_norm = cand_emb / cand_emb.norm(dim=1, keepdim=True)
                        ref_norm = ref_emb / ref_emb.norm(dim=1, keepdim=True)
                         
                        sim_matrix = torch.mm(cand_norm, ref_norm.t())
                         
                        P = sim_matrix.max(dim=1)[0].mean().item()
                        R = sim_matrix.max(dim=0)[0].mean().item()
                        F1 = 2 * (P * R) / (P + R + 1e-10)
                        
                        P_scores.append(P)
                        R_scores.append(R)
                        F1_scores.append(F1)
                     
                    if device == "cuda":
                        torch.cuda.empty_cache()
                
            else:
                for cand, ref in zip(cands, refs):
                    cand_tokens = tokenizer(
                        cand, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                    
                    ref_tokens = tokenizer(
                        ref, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                    
                    cand_emb = model(**cand_tokens).last_hidden_state[0]  
                    ref_emb = model(**ref_tokens).last_hidden_state[0]
                    
                    cand_input_ids = cand_tokens['input_ids'][0]
                    ref_input_ids = ref_tokens['input_ids'][0]
                    
                    cand_mask = (cand_input_ids != tokenizer.cls_token_id) & \
                                (cand_input_ids != tokenizer.sep_token_id) & \
                                (cand_input_ids != tokenizer.pad_token_id) & \
                                (cand_input_ids != tokenizer.mask_token_id) & \
                                (cand_input_ids != tokenizer.unk_token_id)
                    
                    ref_mask = (ref_input_ids != tokenizer.cls_token_id) & \
                               (ref_input_ids != tokenizer.sep_token_id) & \
                               (ref_input_ids != tokenizer.pad_token_id) & \
                               (ref_input_ids != tokenizer.mask_token_id) & \
                               (ref_input_ids != tokenizer.unk_token_id)
                    
                    cand_emb = cand_emb[cand_mask]
                    ref_emb = ref_emb[ref_mask]
                    
                    if cand_emb.size(0) == 0 or ref_emb.size(0) == 0:
                        P_scores.append(0.0)
                        R_scores.append(0.0)
                        F1_scores.append(0.0)
                        continue
                    
                    cand_norm = cand_emb / cand_emb.norm(dim=1, keepdim=True)
                    ref_norm = ref_emb / ref_emb.norm(dim=1, keepdim=True)
                    
                    sim_matrix = torch.mm(cand_norm, ref_norm.t())
                    
                    P = sim_matrix.max(dim=1)[0].mean().item()
                    R = sim_matrix.max(dim=0)[0].mean().item()
                    F1 = 2 * (P * R) / (P + R + 1e-10)
                    
                    P_scores.append(P)
                    R_scores.append(R)
                    F1_scores.append(F1)
                    
        except Exception as e:
            logger.error(f"Error in computing Precision, Recall and F1: {e}")
    
    return np.array(P_scores), np.array(R_scores), np.array(F1_scores)

