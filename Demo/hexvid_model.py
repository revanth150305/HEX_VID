"""HEX-VID inference model, reconstructed from the checkpoint's state-dict layout
(CLIP ViT-B/16 -> Mamba + windowed transformer -> gated fusion -> concept
bottleneck -> hierarchical group/sub-class heads)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel

GROUPS = ["Normal", "Violence", "Theft", "Hazard", "Disorder"]
# Order read off the sub-class axis of results/confusion_matrices.png.
SUBCLASSES = ["Normal", "Assault", "Fighting", "Shooting", "Arson", "Burglary", "Robbery", "Shoplifting",
              "Stealing", "Explosion", "RoadAccidents", "Abuse", "Vandalism", "Arrest"]
CONCEPTS = ["normal_activity", "aggression", "concealment", "speed_motion", "urban_environment",
            "indoor_scene", "outdoor_scene", "night_scene", "group_conflict", "suspicious_movement",
            "property_damage", "uniform_officer", "camera_shake", "physical_contact", "object_transfer",
            "fire_smoke", "vehicle_crash", "crowd_panic", "weapon_visible", "person_running"]  # order read off results/explainability_*.png (bottom to top)


class ClipEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        vm = CLIPVisionModel(CLIPVisionConfig(
            hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
            image_size=224, patch_size=16, hidden_act="quick_gelu"))
        self.vision_model = getattr(vm, "vision_model", vm)  # transformers>=5 drops the wrapper
        self.projection = nn.Linear(768, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):  # (N,3,224,224) -> (N,512)
        return self.layer_norm(self.projection(self.vision_model(pixel_values=x).pooler_output))


class MambaLayer(nn.Module):
    def __init__(self, d=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        di = d * expand
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, di * 2, bias=False)
        self.conv1d = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.delta_proj = nn.Linear(di, di)
        self.B_proj = nn.Linear(di, d_state, bias=False)
        self.C_proj = nn.Linear(di, d_state, bias=False)
        self.A_log = nn.Parameter(torch.zeros(di, d_state))
        self.D_param = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)

    def forward(self, x):
        L = x.shape[1]
        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2))
        delta = F.softplus(self.delta_proj(u))
        Bm, Cm = self.B_proj(u), self.C_proj(u)
        A = -torch.exp(self.A_log)
        h = torch.zeros(u.shape[0], u.shape[2], A.shape[1], device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(delta[:, t, :, None] * A)
            dBu = delta[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            h = dA * h + dBu
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + u * self.D_param
        return x + self.out_proj(y * F.silu(z))


class MambaEncoder(nn.Module):
    def __init__(self, d=512, layers=4):
        super().__init__()
        self.layers = nn.ModuleList(MambaLayer(d) for _ in range(layers))
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.final_norm(x)


class WindowAttn(nn.Module):
    def __init__(self, d=512, heads=4, window=8):
        super().__init__()
        self.h, self.w = heads, window
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        pad = (-T) % self.w
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        n = x.shape[1] // self.w
        qkv = self.qkv(x).view(B * n, self.w, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        o = o.transpose(1, 2).reshape(B, n * self.w, D)
        return self.out_proj(o)[:, :T]


class Block(nn.Module):
    def __init__(self, d=512, heads=4, window=8, ffn=2048):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = WindowAttn(d, heads, window)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(0.1), nn.Linear(ffn, d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _PosEnc(nn.Module):
    def __init__(self, d=512, max_len=512):
        super().__init__()
        pe = torch.zeros(1, max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class WindowedTransformer(nn.Module):
    def __init__(self, d=512, layers=2):
        super().__init__()
        self.pos_enc = _PosEnc(d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(layers))
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.pos_enc(x)
        for b in self.blocks:
            x = b(x)
        return self.final_norm(x)


class GatedFusion(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.gate_proj = nn.Linear(d * 2, d)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, a, b):
        g = torch.sigmoid(self.gate_proj(torch.cat([a, b], -1)))
        return self.layer_norm(g * a + (1 - g) * b)


class TemporalAttention(nn.Module):
    def __init__(self, c=20, heads=4):
        super().__init__()
        self.pool_query = nn.Parameter(torch.zeros(1, 1, c))
        self.self_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(c)
        self.pool_proj = nn.Linear(c, c)

    def forward(self, x):
        x = self.attn_norm(x + self.self_attn(x, x, x, need_weights=False)[0])
        w = torch.softmax((x @ self.pool_query.transpose(1, 2)).squeeze(-1) / math.sqrt(x.shape[-1]), -1)
        return self.pool_proj((w.unsqueeze(-1) * x).sum(1)), w


class ConceptBottleneck(nn.Module):
    def __init__(self, d=512, c=20):
        super().__init__()
        self.concept_predictor = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c))
        self.temporal_attention = TemporalAttention(c)

    def forward(self, x):
        per_frame = self.concept_predictor(x)
        pooled, w = self.temporal_attention(per_frame)
        return pooled, per_frame, w


class Heads(nn.Module):
    def __init__(self, c=20, groups=5, subs=14):
        super().__init__()
        self.group_head, self.sub_head = nn.Linear(c, groups), nn.Linear(c, subs)


class HEXVID(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_encoder = ClipEncoder()
        self.mamba_encoder = MambaEncoder()
        self.transformer_encoder = WindowedTransformer()
        self.gated_fusion = GatedFusion()
        self.concept_bottleneck = ConceptBottleneck()
        self.classification_head = Heads()

    @torch.no_grad()
    def forward(self, frames):  # (B,T,3,224,224)
        B, T = frames.shape[:2]
        f = self.clip_encoder(frames.flatten(0, 1)).view(B, T, -1)
        fused = self.gated_fusion(self.mamba_encoder(f), self.transformer_encoder(f))
        concepts, per_frame, w = self.concept_bottleneck(fused)
        return dict(group=self.classification_head.group_head(concepts),
                    sub=self.classification_head.sub_head(concepts),
                    concepts=concepts, frame_concepts=per_frame, attention=w)


def load_model(path, device="cpu"):
    m = HEXVID()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.to(device).eval(), missing, unexpected
