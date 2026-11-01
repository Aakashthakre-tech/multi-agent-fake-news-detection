import sys
import json
import argparse
from pathlib import Path
import torch

from model import TemporalVideoClassifier, get_feature_extractor
from dataset import sample_video_frames

def predict_video(video_path: Path, checkpoint_path: Path, device: torch.device):
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {"num_frames": 12, "embed_dim": 1280, "hidden_dim": 256, "mlp_dim": 256, "dropout": 0.35})
    class_map = ckpt.get("class_map", {"real": 0, "fake": 1})
    inv_map = {v: k for k, v in class_map.items()}

    # Sample frames
    frames = sample_video_frames(video_path, num_frames=config["num_frames"])
    frames = frames.to(device)

    # Feature extraction
    extractor = get_feature_extractor(device)
    with torch.no_grad():
        features = extractor(frames).unsqueeze(0)  # (1, T, 1280)

    # Classifier
    model = TemporalVideoClassifier(
        num_classes=len(class_map),
        embed_dim=config.get("embed_dim", 1280),
        hidden_dim=config.get("hidden_dim", 256),
        mlp_dim=config.get("mlp_dim", 256),
        dropout=config.get("dropout", 0.35),
        pooling_mode=config.get("pooling_mode", "hybrid"),
        proj_dim=config.get("proj_dim", None),
        norm_type=config.get("norm_type", "none"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(torch.argmax(logits, dim=1).item())

    pred_label = inv_map[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "video": str(video_path.resolve()),
        "predicted_label": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {inv_map[i]: round(float(probs[i]), 4) for i in range(len(probs))},
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single video using TruthLensAI Baseline Classifier")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--checkpoint", type=str, default="video_module/baseline_classifier/best_model.pt", help="Path to best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = predict_video(Path(args.video_path), Path(args.checkpoint), device)
    print(json.dumps(res, indent=2))
