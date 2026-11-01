import torch, os, psutil, gc

ram = psutil.virtual_memory()
print(f"RAM: available={ram.available/1e9:.2f}GB, used={ram.used/1e9:.2f}GB, pct={ram.percent}%")

partial_path = "video_module/baseline_classifier/videomae_train_partial.pt"
cache_path = "video_module/baseline_classifier/videomae_features_cache.pt"

print(f"\nPartial exists: {os.path.exists(partial_path)}", end="")
if os.path.exists(partial_path):
    print(f", size={os.path.getsize(partial_path):,} bytes")
    p = torch.load(partial_path, map_location="cpu", weights_only=False)
    n = len(p["cids"])
    n_real = (p["labels"]==0).sum().item()
    n_fake = (p["labels"]==1).sum().item()
    feat = p["features"]
    nan = torch.isnan(feat).any().item()
    inf = torch.isinf(feat).any().item()
    print(f"  Samples: {n} (real={n_real}, fake={n_fake}), shape={feat.shape}")
    print(f"  Validity: nan={nan}, inf={inf}, norm_min={feat.norm(dim=1).min().item():.3f}")
    print(f"  First 3 cids: {p['cids'][:3]}, Last 3: {p['cids'][-3:]}")
    del p, feat
    gc.collect()
else:
    print()

print(f"\nCache exists: {os.path.exists(cache_path)}", end="")
if os.path.exists(cache_path):
    print(f", size={os.path.getsize(cache_path):,} bytes")
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"  Keys: {list(c.keys())}")
    for k in c:
        if isinstance(c[k], dict) and "features" in c[k]:
            f = c[k]["features"]
            n_real_c = (c[k]["labels"]==0).sum().item()
            n_fake_c = (c[k]["labels"]==1).sum().item()
            nan_c = torch.isnan(f).any().item()
            inf_c = torch.isinf(f).any().item()
            print(f"  [{k}] shape={f.shape} real={n_real_c} fake={n_fake_c} nan={nan_c} inf={inf_c}")
    del c
    gc.collect()
else:
    print()

# Check HF cache for model
hf_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--MCG-NJU--videomae-base")
if os.path.exists(hf_dir):
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fns in os.walk(hf_dir) for f in fns)
    print(f"\nHF model cache: {hf_dir} ({total/1e6:.1f} MB)")
    # List safetensors files
    for dp, dn, fns in os.walk(hf_dir):
        for fn in fns:
            if fn.endswith(".safetensors") or fn.endswith(".json"):
                fp = os.path.join(dp, fn)
                print(f"  {fn} ({os.path.getsize(fp)/1e6:.1f} MB)")
else:
    print(f"\nHF model cache NOT found at {hf_dir}")
