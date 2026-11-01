"""scratch/find_unserializable.py - locate the non-JSON-serializable value."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_evidence_pipeline_audit import chunk  # noqa: E402
from evidence_fusion import calculate_source_agreement  # noqa: E402


def find_bad(obj, path="root"):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += find_bad(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += find_bad(v, f"{path}[{i}]")
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        bad.append((path, type(obj).__name__, repr(obj)[:80]))
    return bad


results = [
    chunk(1, "https://www.reuters.com/a", "SUPPORT", 0.9),
    chunk(2, "https://www.bbc.com/b", "RELATED_CONTEXT", 0.2),
]
agreement = calculate_source_agreement(results)
for p, t, r in find_bad(agreement):
    print(f"{t:12s} at {p}  -> {r}")
print("done")
