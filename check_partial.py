import torch, csv, os

# Check train partial - are ALL 80 entries fake? Inspect cids too
partial = torch.load('video_module/baseline_classifier/videomae_train_partial.pt', map_location='cpu', weights_only=False)
labels = partial['labels']
cids = partial['cids']
print(f'Train partial: {len(cids)} samples')
n_real = (labels==0).sum().item()
n_fake = (labels==1).sum().item()
print(f'Labels: real={n_real}, fake={n_fake}')
print(f'First 5 cids: {cids[:5]}')
print(f'Last 5 cids: {cids[-5:]}')

# Cross-reference with train manifest to see where we are 
train_rows = list(csv.DictReader(open('video_module/dataset_audit/train_manifest.csv')))
extracted_set = set(cids)
total_real_train = sum(1 for r in train_rows if r['label']=='real')
total_fake_train = sum(1 for r in train_rows if r['label']=='fake')
print(f'Train manifest: real={total_real_train}, fake={total_fake_train}')

# Check what comes next in the manifest (order-sensitive)
remaining = [r for r in train_rows if r['cascade_id'] not in extracted_set]
print(f'Remaining in manifest: {len(remaining)} rows')
if remaining:
    print(f'Next 3: {[(r["cascade_id"], r["label"]) for r in remaining[:3]]}')

# Spot-check partial labels
print('First 5 partial (cid, label):')
for c, l in list(zip(cids[:5], labels[:5].tolist())):
    print(f'  {c}: label={l} ({"real" if l==0 else "fake"})')
