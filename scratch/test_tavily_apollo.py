import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import execute_tavily_search

q = '"July 20, 1970" "apollo 11 mission" landed humans moon'
print("Searching:", q)
r, err = execute_tavily_search({"query": q, "topic": "general", "max_results": 5})
print("Err:", err)
for res in (r or {}).get("results", []):
    print(" ", res.get("url"), "|", res.get("title"))
