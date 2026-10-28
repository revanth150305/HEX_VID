"""Weakly supervised model and training components."""
import torch, torch.nn as nn
import clip

class CLIPTSAEncoder(nn.Module):
    def __init__(self, clip_name="ViT-B/32", embed_dim=512, num_heads=4, num_layers=2):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(clip_name, device="cpu")
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim*4,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, clip_frames):
        B, T, C, H, W = clip_frames.shape
        img_emb = self.clip_model.encode_image(clip_frames.view(-1, C, H, W))
        img_emb = img_emb.view(B, T, -1)
        ta_emb = self.transformer(img_emb)
        return ta_emb

def mil_ranking_loss(normal_feats, anomaly_feats, margin=0.5, lamda=1e-4):
    score_fn = nn.Linear(normal_feats.size(-1), 1, bias=False)
    normal_scores = score_fn(normal_feats).squeeze(-1)
    anomaly_scores = score_fn(anomaly_feats).squeeze(-1)
    
    normal_bag = torch.max(normal_scores, dim=1)[0]
    anomaly_bag = torch.max(anomaly_scores, dim=1)[0]
    
    loss_rank = torch.relu(margin - anomaly_bag + normal_bag).mean()
    loss_sparse = torch.mean(torch.abs(anomaly_scores))
    loss_smooth = torch.mean(torch.abs(anomaly_scores[:,1:] - anomaly_scores[:,:-1]))
    
    return loss_rank + lamda * (loss_sparse + loss_smooth)