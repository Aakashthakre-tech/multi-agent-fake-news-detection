"""scratch/check_frontend_startup.py - verify the frontend serves and proxies."""
import json
import re
import urllib.request

FRONTEND = "http://localhost:5173"

print("=== frontend dev server ===")
try:
    with urllib.request.urlopen(FRONTEND + "/", timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
        print("  HTTP status      :", r.status)
except Exception as exc:
    print("  FAILED:", exc)
    raise SystemExit(1)

title = re.search(r"<title>(.*?)</title>", html)
print("  has root div     :", 'id="root"' in html)
print("  loads main.tsx   :", "/src/main.tsx" in html)
print("  title            :", title.group(1) if title else "n/a")

print()
print("=== proxy: frontend /health -> backend ===")
try:
    with urllib.request.urlopen(FRONTEND + "/health", timeout=60) as r:
        payload = json.loads(r.read())
    print("  status           :", payload.get("status"))
    print("  modules          :", list((payload.get("modules") or {}).keys()))
except Exception as exc:
    print("  FAILED:", exc)
