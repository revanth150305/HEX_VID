"""Inference routing between the project's trained models."""
import torch.nn.functional as F

def route_supervised_weak(sup_logits, weak_score, sup_thresh=0.85, weak_thresh=0.5):
    sup_prob = F.softmax(sup_logits, dim=1)[:, 1:]
    sup_conf = sup_prob.max(dim=1)[0]
    weak_prob = weak_score.squeeze(-1)
    
    use_sup = sup_conf >= sup_thresh
    use_weak = weak_prob >= weak_thresh
    
    final = torch.where(use_sup, sup_conf,
                 torch.where(use_weak, weak_prob,
                             (sup_conf + weak_prob) / 2.0))
    source = torch.where(use_sup, torch.tensor(0, device=sup_logits.device),
                  torch.where(use_weak, torch.tensor(1, device=sup_logits.device),
                              torch.tensor(2, device=sup_logits.device)))
    return final, source