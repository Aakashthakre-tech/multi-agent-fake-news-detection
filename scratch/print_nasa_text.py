"""scratch/print_nasa_text.py - print the full extracted NASA article text."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import scrape_url

URL = "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind"

text = scrape_url.invoke(URL) if hasattr(scrape_url, "invoke") else scrape_url(URL)
if text.startswith("SCRAPE_FAILED"):
    print(text)
    raise SystemExit(0)

print("=" * 100)
print(f"FULL EXTRACTED TEXT ({len(text)} chars)")
print("=" * 100)
print(text)

print()
print("=" * 100)
print("LANDING-RELATED WORDING PRESENT?")
print("=" * 100)
low = text.lower()
for marker in ("landed", "landing", "touchdown", "touched down", "eagle",
               "tranquility", "step", "walk", "surface", "moon"):
    hits = [m.start() for m in re.finditer(re.escape(marker), low)]
    print(f"  {marker!r:<16} count={len(hits)}")
    for h in hits[:3]:
        print(f"       ...{text[max(0, h-90):h+90]}...")
