"""scratch/add_entailment_score.py - expose entailment_score in the router fact data."""
import pathlib

p = pathlib.Path(__file__).resolve().parent.parent / "multimodal_router.py"
s = p.read_text(encoding="utf-8")

replacements = [
    (
        '            "contradiction_score": final_verdict.get("contradiction_score"),\n',
        '            "contradiction_score": final_verdict.get("contradiction_score"),\n'
        '            "entailment_score": final_verdict.get("entailment_score"),\n',
    ),
    (
        '            "contradiction_score": final_v.get("contradiction_score"),\n',
        '            "contradiction_score": final_v.get("contradiction_score"),\n'
        '            "entailment_score": final_v.get("entailment_score"),\n',
    ),
    (
        '            "contradiction_score": details.get("contradiction_score"),\n',
        '            "contradiction_score": details.get("contradiction_score"),\n'
        '            "entailment_score": details.get("entailment_score"),\n',
    ),
]

for old, new in replacements:
    count = s.count(old)
    s = s.replace(old, new)
    print(f"{count} site(s): {old.strip()[:60]}")

p.write_text(s, encoding="utf-8")
print("total entailment_score entries:", s.count('"entailment_score"'))
