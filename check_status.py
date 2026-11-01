import torch, os, csv

# Check the train manifest row count vs what was extracted
train_rows = list(csv.DictReader(open('video_module/dataset_audit/train_manifest.csv')))
val_rows = list(csv.DictReader(open('video_module/dataset_audit/val_manifest.csv')))
test_rows = list(csv.DictReader(open('video_module/dataset_audit/test_manifest.csv')))
print(f'Train manifest: {len(train_rows)} rows')
print(f'Val manifest: {len(val_rows)} rows')
print(f'Test manifest: {len(test_rows)} rows')

# Check partial train file
partial = torch.load('video_module/baseline_classifier/videomae_train_partial.pt', map_location='cpu', weights_only=False)
n_partial = len(partial['cids'])
n_train = len(train_rows)
print(f'Train partial checkpoint: {n_partial} of {n_train} extracted')
labels = partial['labels']
print(f'  Label distribution: real={(labels==0).sum().item()}, fake={(labels==1).sum().item()}')

# Val features
vcache = torch.load('video_module/baseline_classifier/videomae_features_cache.pt', map_location='cpu', weights_only=False)
val_feat = vcache['val']
print(f'Val features: {val_feat["features"].shape} (expected {len(val_rows)})')
val_labels = val_feat['labels']
print(f'  Label distribution: real={(val_labels==0).sum().item()}, fake={(val_labels==1).sum().item()}')
print(f'  Duration: {val_feat["duration_sec"]}s')
