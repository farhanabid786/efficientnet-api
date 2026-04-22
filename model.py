"""
model.py — DeepfakeEfficientNet architecture
Matches best_model_v3.pth exactly (EfficientNet-B4 + custom head)
"""

import torch
import torch.nn as nn
import timm

DEFAULT_CONFIG = {
    "model_name"   : "efficientnet_b4",
    "pretrained"   : False,       # weights come from checkpoint, not ImageNet
    "dropout1"     : 0.4,
    "dropout2"     : 0.2,
    "hidden_units" : 512,
    "image_size"   : 380,
    "mean"         : [0.485, 0.456, 0.406],
    "std"          : [0.229, 0.224, 0.225],
    "use_amp"      : False,       # always False on CPU (Render has no GPU)
}


class DeepfakeEfficientNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["model_name"],
            pretrained=cfg.get("pretrained", False),
            num_classes=0,
            global_pool="avg",
        )
        feat = self.backbone.num_features  # 1792 for B4

        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout1"]),
            nn.Linear(feat, cfg["hidden_units"]),
            nn.GELU(),
            nn.BatchNorm1d(cfg["hidden_units"]),
            nn.Dropout(cfg["dropout2"]),
            nn.Linear(cfg["hidden_units"], 1),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x)).squeeze(1)


def load_model(checkpoint_path: str) -> tuple:
    """Load model on CPU. Returns (model, threshold)."""
    device = torch.device("cpu")
    model  = DeepfakeEfficientNet(DEFAULT_CONFIG).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    threshold = float(ckpt.get("optimal_threshold", 0.46))
    return model, threshold
