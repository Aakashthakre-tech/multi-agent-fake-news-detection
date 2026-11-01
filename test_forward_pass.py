"""Quick single-video forward pass test to confirm inference works at current RAM levels."""
import gc, time, csv, psutil, torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoImageProcessor, VideoMAEModel
import cv2, numpy as np

MODEL_ID = "MCG-NJU/videomae-base"
_AUDIT_DIR = Path("C:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System/video_module/dataset_audit")

def get_ram():
    return round(psutil.virtual_memory().available / 1e9, 2)

print(f"RAM start: {get_ram()} GB")
processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
model = VideoMAEModel.from_pretrained(MODEL_ID, local_files_only=True)
sd = model.state_dict()
hf_home = Path.home() / ".cache" / "huggingface" / "hub"
st_path = list((hf_home / "models--MCG-NJU--videomae-base").rglob("model.safetensors"))[0]
raw_sd = load_file(str(st_path))
for i in range(model.config.num_hidden_layers):
    q = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
    v = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
    if q in raw_sd: sd[f"encoder.layer.{i}.attention.attention.query.bias"] = raw_sd[q]
    if v in raw_sd: sd[f"encoder.layer.{i}.attention.attention.value.bias"] = raw_sd[v]
    sd[f"encoder.layer.{i}.attention.attention.key.bias"] = torch.zeros(model.config.hidden_size)
model.load_state_dict(sd, strict=False)
model.eval()
del raw_sd, sd
gc.collect()
print(f"RAM after model+bias fix: {get_ram()} GB")

# Pick the first remaining train video (partial has 120, use row 120)
train_rows = list(csv.DictReader(open(_AUDIT_DIR / "train_manifest.csv", encoding="utf-8")))
row = train_rows[120]  # first unextracted
vid_path = Path(row["file_path"])
cid = row["cascade_id"]
lbl = row["label"]
print(f"Test video: {cid} ({lbl}) at {vid_path}")
assert vid_path.exists(), f"Video missing: {vid_path}"

cap = cv2.VideoCapture(str(vid_path))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(0, max(0, total - 1), 16, dtype=int)
frames = []
last = None
for idx in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, f = cap.read()
    if ret and f is not None:
        last = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(last)
    elif last is not None:
        frames.append(last)
cap.release()
frames_np = np.stack(frames)
print(f"Frames: {frames_np.shape}")

inputs = processor(list(frames_np), return_tensors="pt")
pv = inputs["pixel_values"]
print(f"Input shape: {pv.shape}")
print(f"RAM before fwd: {get_ram()} GB")

t0 = time.time()
with torch.inference_mode():
    out = model(pv)
    emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
elapsed = time.time() - t0

print(f"Forward pass: {elapsed:.2f}s, embedding shape={emb.shape}")
print(f"NaN={torch.isnan(emb).any().item()}, Inf={torch.isinf(emb).any().item()}, norm={emb.norm().item():.3f}")
print(f"RAM after fwd: {get_ram()} GB")
print("FORWARD PASS TEST PASSED")
del out, inputs, pv, frames_np, frames, emb
gc.collect()
print(f"RAM after cleanup: {get_ram()} GB")
