"""scratch/audit_working_tree.py - inspect the working tree for accidental changes and secrets."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


print("=" * 78)
print("1. TRACKED FILE MODIFICATIONS")
print("=" * 78)
print(run("git", "status", "--porcelain") or "(none)")

print("=" * 78)
print("2. .env SAFETY")
print("=" * 78)
ignored = run("git", "check-ignore", "-v", ".env").strip()
print("  gitignored:", ignored or "NOT IGNORED")
status = run("git", "status", "--porcelain")
print("  appears in git status:", ".env" in status)
env = ROOT / ".env"
if env.exists():
    keys = []
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_]+)\s*=", line)
        if m:
            keys.append(m.group(1))
    print(f"  .env present, {len(keys)} key(s) defined, values NOT read: {keys}")
    print("  (no values are printed)")

print("=" * 78)
print("3. SECRETS IN THE TRACKED DIFF")
print("=" * 78)
diff = run("git", "diff")
patterns = {
    "openai key": r"\bsk-[A-Za-z0-9]{20,}",
    "groq key": r"\bgsk_[A-Za-z0-9]{20,}",
    "tavily key": r"\btvly-[A-Za-z0-9]{20,}",
    "bearer token": r"Bearer\s+[A-Za-z0-9._\-]{16,}",
    "provider org id": r"\borg_[A-Za-z0-9]{10,}",
    "hardcoded key assignment": r"api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{12,}",
}
found = False
for name, pattern in patterns.items():
    hits = re.findall(pattern, diff)
    if hits:
        found = True
        print(f"  {name}: {len(hits)} occurrence(s)")
if not found:
    print("  none found")

print("=" * 78)
print("4. SECRETS IN UNTRACKED SOURCE FILES (excluding scratch/ and .env)")
print("=" * 78)
found = False
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)
    parts = {p.lower() for p in rel.parts}
    if "scratch" in parts or "node_modules" in parts or "site-packages" in parts:
        continue
    if rel.name == ".env" or rel.suffix not in (".py", ".ts", ".tsx", ".js", ".json", ".md"):
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            # A redaction pattern in the source is not a leaked secret.
            if "redacted" in text and name in ("provider org id", "bearer token"):
                continue
            found = True
            print(f"  {rel}: {name}")
if not found:
    print("  none found")

print("=" * 78)
print("5. UNTRACKED FILES THAT ARE NEW (top level)")
print("=" * 78)
untracked = [
    line for line in status.splitlines() if line.startswith("??")
]
for line in untracked:
    print("  ", line[3:])
print(f"  total untracked entries: {len(untracked)}")
