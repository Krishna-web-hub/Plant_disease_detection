"""Loads trained crop + disease classifiers for inference."""
import json
import os

import timm
import torch
import yaml
from PIL import Image

from training.utils import build_transforms

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "training_config.yaml")


class ModelBundle:
    def __init__(self, config_path=_CONFIG_PATH):
        self.cfg = yaml.safe_load(open(config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, self.eval_tf = build_transforms(self.cfg["img_size"])

        self.crop_model, self.crop_classes = self._load(
            self.cfg["crop_classifier"]["model_name"], self.cfg["crop_classifier"]["output_dir"]
        )

        self.disease_models = {}
        disease_root = self.cfg["disease_classifier"]["output_dir"]
        if os.path.isdir(disease_root):
            for crop in sorted(os.listdir(disease_root)):
                crop_dir = os.path.join(disease_root, crop)
                if os.path.isfile(os.path.join(crop_dir, "best.pt")):
                    self.disease_models[crop] = self._load(
                        self.cfg["disease_classifier"]["model_name"], crop_dir
                    )

    def _load(self, model_name, output_dir):
        with open(os.path.join(output_dir, "class_to_idx.json")) as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        model = timm.create_model(model_name, pretrained=False, num_classes=len(class_to_idx))
        state = torch.load(os.path.join(output_dir, "best.pt"), map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model, idx_to_class

    @torch.no_grad()
    def _predict(self, model, idx_to_class, image: Image.Image):
        tensor = self.eval_tf(image).unsqueeze(0).to(self.device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return idx_to_class[idx], float(probs[idx])

    def predict_crop(self, image: Image.Image):
        return self._predict(self.crop_model, self.crop_classes, image)

    def predict_disease(self, crop: str, image: Image.Image):
        if crop not in self.disease_models:
            raise KeyError(f"No trained disease classifier for crop '{crop}'")
        model, classes = self.disease_models[crop]
        return self._predict(model, classes, image)
