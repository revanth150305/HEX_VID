import os, re, cv2, numpy as np, torch
from torch.utils.data import Dataset
from torchvision import transforms

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".webm"}
FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FRAME_MANIFEST_PREFIX = "@frames:"

def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path)]


def frame_video_prefix(filename):
    """Return the source-video ID from names such as Abuse001_x264_120.png."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    prefix, separator, frame_number = stem.rpartition("_")
    return prefix if separator and frame_number.isdigit() else stem


def parse_frame_manifest(entry):
    """Parse @frames:<relative directory>:<source-video-ID> entries."""
    if not entry.startswith(FRAME_MANIFEST_PREFIX):
        return None
    try:
        relative_dir, video_id = entry[len(FRAME_MANIFEST_PREFIX):].split(":", 1)
    except ValueError as error:
        raise ValueError(f"Invalid frame manifest entry: {entry}") from error
    return relative_dir, video_id

def discover_class_names(root):
    """Return the stable class order from a Train/<class>, Test/<class> tree."""
    split_dirs = [os.path.join(root, name) for name in ("Train", "Test")]
    names = sorted({entry.name for directory in split_dirs if os.path.isdir(directory)
                    for entry in os.scandir(directory) if entry.is_dir()})
    if not names:
        raise FileNotFoundError(f"Expected class folders under {root}/Train and {root}/Test")
    return names

class UCFClipDataset(Dataset):
    def __init__(self, root, split='train', clip_len=16, stride=2,
                 transform=None, data_list=None, max_clips_per_video=None):
        self.root = root
        self.clip_len = clip_len
        self.stride = stride
        self.transform = transform
        self.max_clips_per_video = max_clips_per_video
        self.class_names = discover_class_names(root)
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.frame_sequences = {}
        
        if data_list is None:
            directory = "Train" if split.lower() == "train" else "Test"
            self.video_paths = [
                os.path.relpath(os.path.join(folder, filename), root)
                for folder, _, files in os.walk(os.path.join(root, directory))
                for filename in files
                if os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS
            ]
        else:
            with open(data_list, encoding="utf-8") as f:
                self.video_paths = [line.strip() for line in f if line.strip()]
        
        self.samples = self._make_dataset()
    
    def _make_dataset(self):
        samples = []
        for vp in self.video_paths:
            manifest = parse_frame_manifest(vp)
            is_frame_sequence = False
            if manifest is not None:
                relative_dir, video_id = manifest
                full_path = os.path.join(self.root, relative_dir)
                if not os.path.isdir(full_path):
                    continue
                frame_paths = sorted(
                    [os.path.join(full_path, name) for name in os.listdir(full_path)
                     if os.path.splitext(name)[1].lower() in FRAME_EXTENSIONS
                     and frame_video_prefix(name) == video_id],
                    key=natural_key,
                )
                self.frame_sequences[vp] = frame_paths
                n_frames = len(frame_paths)
                parts = os.path.normpath(relative_dir).split(os.sep)
                class_name = parts[1] if len(parts) >= 2 else ""
                is_frame_sequence = True
            else:
                full_path = os.path.join(self.root, vp)
                if not os.path.exists(full_path):
                    continue
                if os.path.isdir(full_path):
                    frame_paths = sorted(
                        [os.path.join(full_path, name) for name in os.listdir(full_path)
                         if os.path.splitext(name)[1].lower() in FRAME_EXTENSIONS],
                        key=natural_key,
                    )
                    self.frame_sequences[vp] = frame_paths
                    n_frames = len(frame_paths)
                    parts = os.path.normpath(vp).split(os.sep)
                    class_name = parts[1] if len(parts) >= 2 else ""
                    is_frame_sequence = True
                else:
                    cap = cv2.VideoCapture(full_path)
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    parts = os.path.normpath(vp).split(os.sep)
                    class_name = parts[1] if len(parts) >= 2 else os.path.basename(os.path.dirname(full_path))
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            starts = list(range(0, n_frames - self.clip_len + 1, self.stride))
            if self.max_clips_per_video is not None and len(starts) > self.max_clips_per_video:
                indices = np.linspace(0, len(starts) - 1, self.max_clips_per_video, dtype=int)
                starts = [starts[index] for index in indices]
            for s in starts:
                samples.append((vp, s, label, is_frame_sequence))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vp, start, label, is_frame_sequence = self.samples[idx]
        full_path = os.path.join(self.root, vp)
        frames = []
        if is_frame_sequence:
            for frame_path in self.frame_sequences[vp][start:start + self.clip_len]:
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise RuntimeError(f"Could not read frame: {frame_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            cap = cv2.VideoCapture(full_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(self.clip_len):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        
        if len(frames) < self.clip_len:
            frames.extend([frames[-1]] * (self.clip_len - len(frames)))
        
        clip = np.stack(frames)  # (T, H, W, C)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
        
        if self.transform:
            clip = self.transform(clip)
        return clip, label

    @property
    def num_classes(self):
        return len(self.class_names)
