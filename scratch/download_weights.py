import os
import sys
import requests

url = "https://huggingface.co/mo-thecreator/Deepfake-audio-detection/resolve/main/model.safetensors"
dest_dir = r"C:\Users\ASUS\.cache\huggingface\hub\models--mo-thecreator--Deepfake-audio-detection\snapshots\e4d9874b493362149cec96ced85f00b00b1a04c0"
dest_file = os.path.join(dest_dir, "model.safetensors")
temp_file = dest_file + ".tmp"

os.makedirs(dest_dir, exist_ok=True)
print(f"Connecting to {url} ...")
r = requests.get(url, stream=True, timeout=60)
r.raise_for_status()

total = int(r.headers.get("content-length", 0))
downloaded = 0

print(f"Downloading {total / 1024 / 1024:.1f} MB...")
with open(temp_file, "wb") as f:
    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total) * 100 if total else 0
            print(f"Downloaded {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)", flush=True)

if os.path.exists(dest_file):
    os.remove(dest_file)
os.replace(temp_file, dest_file)
print(f"Successfully saved to {dest_file}. Size: {os.path.getsize(dest_file)} bytes.")
