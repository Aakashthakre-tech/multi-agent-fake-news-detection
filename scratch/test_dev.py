# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')
from video_module.claim_ranker import detect_predicate_and_verbs

import re
text = "मुंबईत मोठा अपघात झाला, ५ जणांचा मृत्यू"
words = re.findall(r"[^\s.,!?;:\"\'()\[\]{}—–]+", text.lower())
print("words with punct split:", [repr(w) for w in words])
from video_module.claim_ranker import ALL_DEVANAGARI_VERBS
print("Matches:", [w for w in words if w in ALL_DEVANAGARI_VERBS])
