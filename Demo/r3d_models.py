"""R3D-18 (Kinetics-400 style) video classifiers for the sup_best / mil_best checkpoints."""
import torch
import torch.nn as nn

CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "NormalVideos",
           "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def _conv(i, o, stride):
    return nn.Conv3d(i, o, 3, stride=stride, padding=1, bias=False)


class Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.conv1 = nn.Sequential(_conv(i, o, stride), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(_conv(o, o, 1), nn.BatchNorm3d(o))
        self.downsample = (nn.Sequential(nn.Conv3d(i, o, 1, stride=stride, bias=False), nn.BatchNorm3d(o))
                           if stride != 1 or i != o else None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        return self.relu(self.conv2(self.conv1(x)) + r)


class R3D18(nn.Module):
    def __init__(self, head_name, n=14):
        super().__init__()
        self.backbone = nn.Module()
        b = self.backbone
        b.stem = nn.Sequential(nn.Conv3d(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                               nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        cfg, i = [(64, 1), (128, 2), (256, 2), (512, 2)], 64
        for k, (o, s) in enumerate(cfg, 1):
            setattr(b, f"layer{k}", nn.Sequential(Block(i, o, s), Block(o, o, 1)))
            i = o
        self.head_name = head_name
        setattr(self, head_name, nn.Linear(512, n))

    def forward(self, x):  # (N,3,T,H,W) -> (N,14)
        b = self.backbone
        x = b.layer4(b.layer3(b.layer2(b.layer1(b.stem(x)))))
        return getattr(self, self.head_name)(x.mean((2, 3, 4)))


def load_r3d(path, kind, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = R3D18("classifier" if kind == "sup" else "head")
    m.load_state_dict(ck["state_dict"] if kind == "sup" else ck["model"])
    topk = ck["args"]["topk"] if kind == "mil" else None
    return m.to(device).eval(), topk
