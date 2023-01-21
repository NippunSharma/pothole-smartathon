import torch

config = {
    "detection": {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "conf": 0.25,
        "iou": 0.3,
        "agnostic": False,
        "augment": True,
        "multi_label": False,
        "classes": None,
        "max_det": 20,
        "amp": False
    }
}
