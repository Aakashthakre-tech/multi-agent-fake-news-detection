import csv

train_rows = list(csv.DictReader(open('video_module/dataset_audit/train_manifest.csv')))
print('First 10 rows of train manifest:')
for r in train_rows[:10]:
    cid = r["cascade_id"]
    lbl = r["label"]
    print(f'  {cid}: {lbl}')

labels_seq = [r["label"] for r in train_rows]
first_real = next((i for i, l in enumerate(labels_seq) if l=="real"), None)
first_fake = next((i for i, l in enumerate(labels_seq) if l=="fake"), None)
print(f'\nFirst fake at index: {first_fake}')
print(f'First real at index: {first_real}')

# Count fakes before first real
n_fake_before_real = sum(1 for l in labels_seq[:first_real] if l=="fake")
print(f'Fakes before first real: {n_fake_before_real}')

# Count label distribution in first 80 rows
first_80 = labels_seq[:80]
print(f'First 80 rows: real={first_80.count("real")}, fake={first_80.count("fake")}')

print('\nRows 78-88 of manifest:')
for i, r in enumerate(train_rows[78:88], 78):
    cid = r["cascade_id"]
    lbl = r["label"]
    print(f'  [{i}] {cid}: {lbl}')
