"""scratch/test_tavily.py - Test Tavily API connectivity"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import search_with_diagnostics

result = search_with_diagnostics("The Apollo 11 mission landed humans on the Moon in July 1969.")
print("search_with_diagnostics result keys:", list(result.keys()))
print(f"output length: {len(result.get('output', ''))}")
print(f"plans count: {len(result.get('plans', []))}")
print(f"search_errors count: {len(result.get('search_errors', []))}")
print(f"candidates count: {len(result.get('candidates', []))}")

if result.get('search_errors'):
    print("\nSEARCH ERRORS:")
    for e in result['search_errors']:
        print(f"  - {e}")

if result.get('candidates'):
    print(f"\nFirst candidate:")
    c = result['candidates'][0]
    for k, v in c.items():
        if k != 'content':
            print(f"  {k}: {v}")
        else:
            print(f"  content (first 200 chars): {str(v)[:150]}...")
