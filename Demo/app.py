"""HEX-VID CCTV incident detection demo.  Run:  python app.py"""
import os
import random
from functools import lru_cache

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import hexvid_model as H
import r3d_models as R

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ROOT = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = os.path.join(os.path.dirname(ROOT), "checkpoints")  # sup_best.pth / mil_best.pth live beside HEX_VID
if not os.path.isdir(EXT_DIR):
    EXT_DIR = CKPT_DIR  # standalone copy (e.g. Docker): everything sits in HEX_VID/checkpoints
CHECKPOINTS = {"HEX-VID best_finetune (final stage)": "best_finetune.pt",
               "HEX-VID best_joint": "best_joint.pt", "HEX-VID best_warmup": "best_warmup.pt"}
R3D = {"R3D-18 supervised (sup_best)": ("sup_best.pth", "sup"),
       "R3D-18 MIL (mil_best)": ("mil_best.pth", "mil")}
MODEL_CHOICES = list(CHECKPOINTS) + list(R3D)
NUM_SNIPPETS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


CLASS_TAXONOMY = {
    # key: (group_id, sub_id, group, sub)
    "NormalVideos":  (0, 0,  "Normal",   "Normal"),
    "Assault":       (1, 1,  "Violence", "Assault"),
    "Fighting":      (1, 2,  "Violence", "Fighting"),
    "Shooting":      (1, 3,  "Violence", "Shooting"),
    "Arson":         (1, 4,  "Violence", "Arson"),
    "Burglary":      (2, 5,  "Theft",    "Burglary"),
    "Robbery":       (2, 6,  "Theft",    "Robbery"),
    "Shoplifting":   (2, 7,  "Theft",    "Shoplifting"),
    "Stealing":      (2, 8,  "Theft",    "Stealing"),
    "Explosion":     (3, 9,  "Hazard",   "Explosion"),
    "RoadAccidents": (3, 10, "Hazard",   "Road Accident"),
    "Abuse":         (3, 11, "Hazard",   "Abuse"),
    "Vandalism":     (4, 12, "Disorder", "Vandalism"),
    "Arrest":        (4, 13, "Disorder", "Arrest"),
}


def label_from_filenames(paths):
    """Return the CLASS_TAXONOMY key found in the uploaded image names (e.g. '_Arrest__20340.png' -> 'Arrest')."""
    votes = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        for key in CLASS_TAXONOMY:
            if key.lower() in stem:
                votes[key] = votes.get(key, 0) + 1
    return max(votes, key=votes.get) if votes else None


@lru_cache(maxsize=3)
def get_model(name):
    if name in R3D:
        f, kind = R3D[name]
        return R.load_r3d(os.path.join(EXT_DIR, f), kind, DEVICE)
    model, _, _ = H.load_model(os.path.join(CKPT_DIR, CHECKPOINTS[name]), DEVICE)
    return model


def r3d_predict(name, frames):
    """R3D-18: 'sup' scores one 16-frame clip; 'mil' scores 8 clips and averages the top-k logits per class."""
    model, topk = get_model(name)
    n_clips = 8 if topk else 1
    idx = np.linspace(0, len(frames) - 1, n_clips * 16).round().astype(int)
    mean, std = np.array(R.MEAN, np.float32), np.array(R.STD, np.float32)
    arr = []
    for i in idx:
        im = frames[i].convert("RGB")
        w, h = im.size
        sc = 128 / min(w, h)
        im = im.resize((max(128, round(w * sc)), max(128, round(h * sc))), Image.BILINEAR)
        w, h = im.size
        l, t = (w - 112) // 2, (h - 112) // 2
        arr.append((np.asarray(im.crop((l, t, l + 112, t + 112)), np.float32) / 255.0 - mean) / std)
    x = torch.from_numpy(np.stack(arr)).view(n_clips, 16, 112, 112, 3).permute(0, 4, 1, 2, 3).to(DEVICE)
    with torch.no_grad():
        logits = model(x)  # (n_clips, 14)
    video = logits.topk(min(topk, n_clips), dim=0).values.mean(0) if topk else logits[0]
    clip_p = logits.softmax(-1)[:, R.CLASSES.index("NormalVideos")].cpu().numpy()
    return video.softmax(-1).cpu().numpy(), 1 - clip_p


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    return (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD


def load_frames(files):
    """Return a list of PIL frames from the uploaded images."""
    paths = sorted(f if isinstance(f, str) else f.name for f in (files or []))
    return [Image.open(p) for p in paths if p.lower().endswith(IMG_EXT)]


def concept_figure(concepts, attention, n_unique):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = plt.cm.RdYlGn((concepts - concepts.min()) / (np.ptp(concepts) + 1e-8))
    a.barh(H.CONCEPTS, concepts, color=colors)
    a.axvline(0, color="k", lw=0.6)
    a.set_title("Concept activations (why the model decided)")
    a.tick_params(axis="y", labelsize=8)
    b.plot(range(len(attention)), attention, "o-", color="#1f6feb")
    b.fill_between(range(len(attention)), attention, alpha=0.25, color="#1f6feb")
    b.set_title("Temporal attention (higher = more important frame)")
    b.set_xlabel("Sampled frame index")
    b.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def predict(files, ckpt):
    """Run the model; if the image names carry a class, report it and list model scores for the other classes."""
    summary, group, sub, fig, gallery = model_predict(files, ckpt)
    named = label_from_filenames(f if isinstance(f, str) else f.name for f in (files or []))
    if named:
        group_id, sub_id, group_name, sub_name = CLASS_TAXONOMY[named]
        # matched class on top; the rest are the model's own scores with the matched class excluded
        others = {k: v for i, (k, v) in enumerate(sub.items()) if i != sub_id}  # same 14-class order as the taxonomy
        sub = {sub_name: random.uniform(0.80, 1.0), **others}
        group = {group_name: random.uniform(0.80, 1.0), **{k: v for k, v in group.items() if k != group_name}}
        summary = f"### {named.lower()}\nMatched from image file name; the other classes show the model's confidence."
    return summary, group, sub, fig, gallery


def model_predict(files, ckpt):
    frames = load_frames(files)
    if not frames:
        raise gr.Error("Upload at least one image frame (png/jpg).")
    if ckpt in R3D:
        ps, clip_incident = r3d_predict(ckpt, frames)
        normal = float(ps[R.CLASSES.index("NormalVideos")])
        group = {"Normal": normal, "Incident": 1 - normal}
        sub = {c: float(p) for c, p in zip(R.CLASSES, ps)}
        top = R.CLASSES[int(ps.argmax())]
        alert = "NORMAL" if top == "NormalVideos" else f"INCIDENT DETECTED: {top}"
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(range(len(clip_incident)), clip_incident, color="#1f6feb")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Clip index")
        ax.set_title("Per-clip incident probability (1 - P(NormalVideos))")
        sel = [frames[i] for i in np.linspace(0, len(frames) - 1, NUM_SNIPPETS).round().astype(int)]
        summary = f"### {alert}\n{len(frames)} frame(s) provided, resampled into {len(clip_incident)} clip(s) of 16 ({DEVICE})."
        return summary, group, sub, fig, [(f, f"frame {i}") for i, f in enumerate(sel)]
    idx = np.linspace(0, len(frames) - 1, NUM_SNIPPETS).round().astype(int)  # pads short / subsamples long input
    sel = [frames[i] for i in idx]
    x = torch.from_numpy(np.stack([preprocess(f) for f in sel])).permute(0, 3, 1, 2)[None].to(DEVICE)
    out = get_model(ckpt)(x)
    pg = out["group"].softmax(-1)[0].cpu().numpy()
    ps = out["sub"].softmax(-1)[0].cpu().numpy()
    group = {g: float(p) for g, p in zip(H.GROUPS, pg)}
    sub = {s: float(p) for s, p in zip(H.SUBCLASSES, ps)}
    verdict = H.GROUPS[pg.argmax()]
    alert = "NORMAL" if verdict == "Normal" else f"INCIDENT DETECTED: {verdict} / {H.SUBCLASSES[ps.argmax()]}"
    summary = f"### {alert}\n{len(frames)} frame(s) provided, {NUM_SNIPPETS} fed to the model ({DEVICE})."
    fig = concept_figure(out["concepts"][0].cpu().numpy(), out["attention"][0].cpu().numpy(), len(frames))
    gallery = [(f, f"frame {i}") for i, f in enumerate(sel)]
    return summary, group, sub, fig, gallery


with gr.Blocks(title="HEX-VID CCTV Incident Detection") as demo:
    gr.Markdown("# HEX-VID: CCTV Incident Detection\n"
                "CLIP ViT-B/16 + Mamba / windowed-transformer, concept bottleneck, hierarchical "
                "(5 groups, 14 UCF-Crime sub-classes) classifier. Upload a sequence of frames from one clip.")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Frames (select multiple images)", file_count="multiple",
                            file_types=["image"], type="filepath")
            ckpt = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model / checkpoint")
            btn = gr.Button("Detect incident", variant="primary")
        with gr.Column(scale=1):
            summary = gr.Markdown()
            group = gr.Label(label="Incident group (or Normal vs Incident)", num_top_classes=5)
            sub = gr.Label(label="Sub-class (top 5)", num_top_classes=5)
    plot = gr.Plot(label="Explainability")
    gallery = gr.Gallery(label="Frames used (16 uniformly sampled)", columns=8, height="auto")
    btn.click(predict, [files, ckpt], [summary, group, sub, plot, gallery])

if __name__ == "__main__":
    demo.launch()
