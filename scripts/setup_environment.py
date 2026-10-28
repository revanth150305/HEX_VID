import os
import kagglehub
import random
import argparse
from collections import defaultdict
import sys

sys.path.insert(0, "src")
from datasets import FRAME_EXTENSIONS, frame_video_prefix

def setup_environment(data_root=None):
    """Download dataset and create stratified subset"""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/subset", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Colab can use a KaggleHub cache directly. This avoids copying a large
    # dataset through the mounted Google Drive filesystem.
    if data_root is not None:
        data_root = os.path.abspath(data_root)
        if not (os.path.isdir(os.path.join(data_root, "Train")) and
                os.path.isdir(os.path.join(data_root, "Test"))):
            raise FileNotFoundError(
                f"Expected Train and Test folders under --data-root: {data_root}"
            )
        create_lists_from_folder_layout(data_root)
        return

    # A manually supplied dataset already follows the supported layout; do not
    # download another 11 GB copy or require the original split-list files.
    if os.path.isdir("data/raw/ucf_crime/Train") and os.path.isdir("data/raw/ucf_crime/Test"):
        create_lists_from_folder_layout()
        return
    
    # Download UCF-Crime dataset
    print("Downloading UCF-Crime dataset...")
    path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Symlink to our structure
    if not os.path.exists("data/raw/ucf_crime"):
        os.symlink(path, "data/raw/ucf_crime", target_is_directory=True)
    
    # Create stratified subset
    create_stratified_subset()

def create_lists_from_folder_layout(root="data/raw/ucf_crime"):
    """Create train/test lists for a root containing Train/<class> and Test/<class>."""
    out_dir = "data/subset"
    video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".webm"}

    def videos_in(split, class_name=None):
        start = os.path.join(root, split, class_name) if class_name else os.path.join(root, split)
        found = []
        for folder, _, filenames in os.walk(start):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in video_extensions:
                    found.append(os.path.relpath(os.path.join(folder, filename), root).replace("\\", "/"))
        # Kaggle's PNG files include original video IDs, for example
        # Abuse001_x264_120.png. Make one manifest entry per original video
        # rather than treating the whole class directory as one giant video.
        for folder, _, filenames in os.walk(start):
            video_ids = sorted({
                frame_video_prefix(filename)
                for filename in filenames
                if os.path.splitext(filename)[1].lower() in FRAME_EXTENSIONS
            })
            relative_dir = os.path.relpath(folder, root).replace("\\", "/")
            found.extend(f"@frames:{relative_dir}:{video_id}" for video_id in video_ids)
        return found

    classes = sorted(name for name in os.listdir(os.path.join(root, "Train"))
                     if os.path.isdir(os.path.join(root, "Train", name)))
    if not classes:
        raise RuntimeError("No class folders found in data/raw/ucf_crime/Train")

    random.seed(42)
    train_videos = []
    for class_name in classes:
        class_videos = videos_in("Train", class_name)
        random.shuffle(class_videos)
        train_videos.extend(class_videos[:30])
    test_videos = videos_in("Test")
    random.shuffle(test_videos)
    test_videos = test_videos[:200]
    if not train_videos or not test_videos:
        raise RuntimeError("No supported video files or PNG/JPG frame sequences found under Train and Test.")

    with open(os.path.join(out_dir, "ucf_crime_train_subset.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_videos))
    with open(os.path.join(out_dir, "ucf_crime_test_subset.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_videos))
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"Train subset: {len(train_videos)} videos")
    print(f"Test subset: {len(test_videos)} videos")

def create_stratified_subset():
    """Create stratified subset of the dataset"""
    
    UCF_ROOT = "data/raw/ucf_crime"
    OUT_DIR = "data/subset"
    
    def load_list(txt_path):
        with open(txt_path) as f:
            return [line.strip() for line in f if line.strip()]
    
    # Load original split files
    train_vids = load_list(os.path.join(UCF_ROOT, "Train_Test_Train.txt"))
    test_vids = load_list(os.path.join(UCF_ROOT, "Train_Test_Test.txt"))
    
    def ucf_class(p):
        return p.split('/')[0].split('_')[0]
    
    # Stratify by class
    cls_to_vids = defaultdict(list)
    for v in train_vids:
        cls_to_vids[ucf_class(v)].append(v)
    
    # Budget per class
    budget = {
        "Normal":150, "Abuse":30, "Arrest":30, "Burglary":30,
        "Explosion":20, "Fighting":30, "RoadAccidents":20,
        "Robbery":20, "Shooting":20, "Shoplifting":20,
        "Stealing":20, "Vandalism":20
    }
    
    subset_train = []
    for cls, n in budget.items():
        avail = cls_to_vids.get(cls, [])
        random.shuffle(avail)
        subset_train.extend(avail[:n])
    
    # Same for test
    subset_test = random.sample(test_vids, min(200, len(test_vids)))
    
    # Save subset lists
    with open(os.path.join(OUT_DIR, "ucf_crime_train_subset.txt"), "w") as f:
        f.write("\n".join(subset_train))
    with open(os.path.join(OUT_DIR, "ucf_crime_test_subset.txt"), "w") as f:
        f.write("\n".join(subset_test))
    
    print(f"Train subset: {len(subset_train)} videos")
    print(f"Test subset:  {len(subset_test)} videos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create UCF-Crime subset lists.")
    parser.add_argument(
        "--data-root",
        help="Existing root containing Train and Test; useful for a Colab KaggleHub cache.",
    )
    setup_environment(parser.parse_args().data_root)
