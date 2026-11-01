from pathlib import Path

path = Path("claim_profile.py")
content = path.read_text(encoding="utf-8")

old_sort = "phrases.sort(key=lambda p: (-len(p.split()), -len(p)))"
new_sort = "phrases.sort(key=lambda p: (-any(c.isdigit() for c in p), -len(p.split()), -len(p)))"

assert old_sort in content, "old_sort not found in claim_profile.py"
content = content.replace(old_sort, new_sort)
path.write_text(content, encoding="utf-8")
print("Successfully patched phrases sort in claim_profile.py")
