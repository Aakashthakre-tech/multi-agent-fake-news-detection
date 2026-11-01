import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any, Optional

# Standard ImageNet normalization for EfficientNet-B0
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def sample_video_frames(
    video_path: Path,
    num_frames: int = 12,
    transform: Optional[transforms.Compose] = None,
) -> torch.Tensor:
    """
    Uniformly sample `num_frames` from a video file and return a Tensor of shape (num_frames, 3, 224, 224).
    Handles variable video durations, short clips, and invalid frame reads safely.
    """
    if transform is None:
        transform = DEFAULT_TRANSFORM

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not (fps == fps):
            fps = 25.0

        if total_frames <= 0:
            # Fallback: scan through frames manually if total_frames property is unavailable
            frames_list = []
            while len(frames_list) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_list.append(frame)
            if not frames_list:
                raise ValueError(f"No decodable frames found in {video_path}")
            # Pad by repeating last frame if necessary
            while len(frames_list) < num_frames:
                frames_list.append(frames_list[-1])
            frame_indices = list(range(num_frames))
        else:
            # Uniformly space target frame indices across the duration
            frame_indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)

        sampled_tensors = []
        last_valid_tensor = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                tensor_img = transform(pil_img)
                last_valid_tensor = tensor_img
                sampled_tensors.append(tensor_img)
            else:
                # Fallback to last valid frame or neutral zero-frame
                if last_valid_tensor is not None:
                    sampled_tensors.append(last_valid_tensor)
                else:
                    # If initial frame fails, read sequentially from start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_alt, frame_alt = cap.read()
                    if ret_alt and frame_alt is not None:
                        rgb_alt = cv2.cvtColor(frame_alt, cv2.COLOR_BGR2RGB)
                        last_valid_tensor = transform(Image.fromarray(rgb_alt))
                        sampled_tensors.append(last_valid_tensor)
                    else:
                        # Absolute fallback: neutral black frame
                        black_img = Image.new("RGB", (224, 224), (0, 0, 0))
                        last_valid_tensor = transform(black_img)
                        sampled_tensors.append(last_valid_tensor)

        # Ensure exactly num_frames are returned
        while len(sampled_tensors) < num_frames:
            sampled_tensors.append(last_valid_tensor)

        sampled_tensors = sampled_tensors[:num_frames]
        # Stack to (num_frames, 3, 224, 224)
        return torch.stack(sampled_tensors, dim=0)

    finally:
        cap.release()
