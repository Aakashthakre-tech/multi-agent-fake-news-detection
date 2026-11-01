"""scratch/probe_nli_labels.py - verify the NLI model's real id2label mapping."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nli_checker import DEFAULT_LABELS, LABEL_ALIASES, check_nli

print("=" * 84)
print("MODEL LABEL SET")
print("=" * 84)
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("cross-encoder/nli-deberta-v3-base")
    print(f"  model id2label : {cfg.id2label}")
    print(f"  code DEFAULT_LABELS : {DEFAULT_LABELS}")
    print(f"  code LABEL_ALIASES  :")
    for k, v in LABEL_ALIASES.items():
        print(f"      {k:<20} -> {v}")
except Exception as exc:
    print(f"  could not load config: {exc}")

print()
print("=" * 84)
print("LIVE NLI ON A RELEVANT PASSAGE (real model, real call)")
print("=" * 84)
CLAIM = "NASA's Apollo 11 mission landed humans on the Moon in July 1969."
PASSAGES = [
    ("NASA page-style",
     "Apollo 11 was the mission that landed the first humans on the Moon. "
     "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle in the "
     "Sea of Tranquility on July 20, 1969."),
    ("Smithsonian-style",
     "Apollo 11: The first steps on the Moon. On July 20, 1969, Apollo 11 "
     "landed on the Moon and Neil Armstrong and Edwin 'Buzz' Aldrin became "
     "the first humans to walk on the lunar surface."),
    ("irrelevant",
     "The Apollo program was cancelled in the 1970s; the Saturn V rocket was "
     "used for the Skylab and Apollo-Soyuz missions."),
]
for label, text in PASSAGES:
    r = check_nli(CLAIM, text)
    print(f"  [{label}]")
    print(f"      label  = {r['label']}  score = {r['score']}  status = {r['status']}")
    print(f"      scores = {r.get('scores')}")
