"""scratch/probe_retry.py - does langchain_core's retry respect Retry-After?"""
import importlib
import inspect

for mod in (
    "langchain_core.utils.retry",
    "langchain_core.language_models.base",
    "langchain_core.language_models.chat_models",
    "langchain_core.runnables.base",
):
    try:
        m = importlib.import_module(mod)
    except Exception as exc:
        print(f"{mod}: cannot import ({exc})")
        continue
    for name in dir(m):
        if "retry" not in name.lower():
            continue
        obj = getattr(m, name)
        if not callable(obj):
            continue
        try:
            print(f"{mod}.{name}{inspect.signature(obj)}")
        except Exception:
            print(f"{mod}.{name}  (signature unavailable)")

print()
print("--- ChatGroq default max_retries ---")
try:
    from langchain_groq import ChatGroq
    print(inspect.signature(ChatGroq).parameters.get("max_retries"))
except Exception as exc:
    print(exc)
