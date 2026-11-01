import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from evidence_analyzer import split_into_chunks, _build_context_windows
from test_evidence_retrieval_reliability import CLAIM, NASA_ARTICLE, NASA_PARAGRAPHS

chunks = split_into_chunks(NASA_ARTICLE)
print(f"NASA_ARTICLE has {len(NASA_ARTICLE)} chars")
print(f"split_into_chunks produces {len(chunks)} chunks")
print()
for i, ch in enumerate(chunks):
    print(f"=== chunk[{i}] ({len(ch)} chars) ===")
    print(repr(ch[:80]))
    print()

# Check if windows are substrings
windows = _build_context_windows(chunks, list(range(len(chunks))), 10, 300)
print(f"\nTotal windows built: {len(windows)}")
for w in windows:
    if "Eagle" in w or "first human foot" in w:
        in_article = w.strip() in NASA_ARTICLE
        print(f"Window with 'Eagle' in NASA_ARTICLE: {in_article}")
        print("Window text:", repr(w[:200]))
        print()
