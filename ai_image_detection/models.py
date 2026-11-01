"""
models.py
=========
Uniform adapters for candidate AI-image detectors.

Every adapter exposes the same contract:

    adapter.predict_probability(pil_image) -> float
        Probability that the image is AI-GENERATED, in [0, 1].

That single number is all the benchmark needs. Thresholding, calibration and
ensembling all operate on it afterwards, so adding a new model never requires
changing evaluation code.

Adapters never hardcode anything about a specific image. They receive a PIL
image object and return a number produced by a model forward pass.
"""

import torch
from PIL import Image


def _pil_to_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


class TransformersImageClassifier:
    """
    Adapter for a transformers image-classification checkpoint that separates
    AI-generated images from real ones.

    The mapping from the checkpoint's own label vocabulary to
    (ai_probability, real_probability) is discovered from the config, the same
    way the original detector.py does it, rather than assumed.
    """

    def __init__(self, name, prefer_label="ai", device="cpu"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.name = name
        self.device = device
        self.prefer_label = prefer_label

        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name)
        self.model.eval()
        self.model.to(device)

        self.id2label = self.model.config.id2label
        self._resolve_labels()

    def _resolve_labels(self):
        """
        Find which logit index means AI and which means real.

        Falls back to index 0 / 1 when the vocabulary is uninformative, which
        is recorded in self.label_mapping_source so a report can say so.
        """
        labels = {i: str(v).lower() for i, v in self.id2label.items()}

        ai_idx, real_idx = None, None

        for i, lab in labels.items():
            if any(k in lab for k in ("fake", "ai", "generated", "synthetic",
                                      "artificial")):
                ai_idx = i
            elif any(k in lab for k in ("real", "human", "nature")):
                real_idx = i

        if ai_idx is None or real_idx is None:
            self.ai_index = 0
            self.real_index = 1 if len(labels) > 1 else 0
            self.label_mapping_source = "positional_fallback"
        else:
            self.ai_index = ai_idx
            self.real_index = real_idx
            self.label_mapping_source = "from_config"

    def predict_probability(self, image):
        image = _pil_to_rgb(image)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1)[0]

        return float(probs[self.ai_index].item())

    def describe(self):
        return {
            "name": self.name,
            "type": "transformers_classifier",
            "id2label": {str(k): str(v) for k, v in self.id2label.items()},
            "ai_index": self.ai_index,
            "real_index": self.real_index,
            "label_mapping_source": self.label_mapping_source,
        }


class ClipZeroShotDetector:
    """
    Adapter that asks a CLIP model directly whether the image looks like a
    photograph or like a generated picture.

    This is a deliberately different mechanism from the supervised detectors:
    no forensics, just a semantic judgement from image-text alignment. It is
    included as a candidate because a different failure mode may not be
    correlated with the others, which is the only thing that would justify an
    ensemble later.
    """

    AI_PROMPT = "an AI-generated image"
    REAL_PROMPT = "a real photograph"

    def __init__(self, name="openai/clip-vit-base-patch32", device="cpu"):
        from transformers import CLIPModel, CLIPProcessor

        self.name = name
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name)
        self.model.eval()
        self.model.to(device)

    def predict_probability(self, image):
        image = _pil_to_rgb(image)
        inputs = self.processor(
            text=[self.AI_PROMPT, self.REAL_PROMPT],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Compare each prompt against the image, then softmax.
        #
        # The learned logit_scale MUST be applied. CLIP's raw cosine
        # similarities sit in a narrow band near 1.0, so a softmax over the
        # bare similarities returns ~0.5 for every image and the detector
        # carries no signal at all. The model is trained to use this scale.
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        sims = logit_scale * (image_embeds @ text_embeds.T)[0]
        probs = torch.softmax(sims, dim=0)

        return float(probs[0].item())

    def describe(self):
        return {
            "name": self.name,
            "type": "clip_zero_shot",
            "prompts": [self.AI_PROMPT, self.REAL_PROMPT],
        }


# Registry of candidate models. The benchmark iterates this mapping, so every
# candidate is measured on exactly the same images.
CANDIDATES = {
    "steganalysis_current": lambda: TransformersImageClassifier(
        "delpot/steganograph-ia-detector"),
    "deepfake_ai_detector": lambda: TransformersImageClassifier(
        "prithivMLmods/Deep-Fake-Detector-Model"),
    "ai_art_detector": lambda: TransformersImageClassifier(
        "umm-maybe/AI-image-detector"),
    "clip_zeroshot": lambda: ClipZeroShotDetector(
        "openai/clip-vit-base-patch32"),
}


def build_adapter(key):
    """Instantiate one candidate adapter by registry key."""
    if key not in CANDIDATES:
        raise KeyError(
            "Unknown candidate %r. Available: %s"
            % (key, sorted(CANDIDATES))
        )
    return CANDIDATES[key]()
