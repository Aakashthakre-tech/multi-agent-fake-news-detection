"""scratch/replace_test_identifiers.py - make the test fixtures unmistakably synthetic."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The real account identifiers must not sit in the repository, even in tests.
# These placeholders are deliberately not the shape of a real id.
REPLACEMENTS = {
    "org_01kwyrbkeqf58a61y60rnaa7jx": "org_01EXAMPLEPLACEHOLDER00",
    "gsk_abcdefghijklmnopqrstuv": "gsk_PLACEHOLDERKEYPLACEHOLDER",
}

targets = [
    ROOT / "tests" / "test_policy_diagnostics_propagation.py",
    ROOT / "tests" / "test_text_evaluation_set.py",
]

for path in targets:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"updated {path.name}")
    else:
        print(f"no change in {path.name}")

# Verify nothing of the old shape remains anywhere in the source tree.
print()
print("verification:")
for path in ROOT.rglob("*.py"):
    if "scratch" in {p.lower() for p in path.relative_to(ROOT).parts}:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    for old in REPLACEMENTS:
        if old in text:
            print(f"  STILL PRESENT in {path.relative_to(ROOT)}: {old}")
print("  done")
