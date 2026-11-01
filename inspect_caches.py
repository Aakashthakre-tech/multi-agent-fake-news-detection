import torch, os

def describe(v, indent=2):
    sp = " " * indent
    if isinstance(v, torch.Tensor):
        print(f"{sp}Tensor shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, dict):
        print(f"{sp}dict with {len(v)} keys")
        for k2, v2 in list(v.items())[:5]:
            print(f"{sp}  [{repr(k2)[:60]}]:")
            describe(v2, indent+4)
    elif isinstance(v, list):
        print(f"{sp}list len={len(v)}")
    else:
        print(f"{sp}{type(v).__name__}: {repr(v)[:80]}")

for fname in [
    "video_module/baseline_classifier/videomae_features_cache.pt",
    "video_module/baseline_classifier/videomae_train_partial.pt",
    "video_module/baseline_classifier/features_cache.pt",
]:
    print(f"\n=== {fname} ({os.path.getsize(fname):,} bytes) ===")
    try:
        d = torch.load(fname, map_location="cpu", weights_only=False)
        describe(d, indent=2)
    except Exception as e:
        print(f"  ERROR: {e}")
