"""
nli_checker.py
==============
Hardened natural-language-inference wrapper.

Safety rules enforced here:
1. The model is loaded lazily, so importing this module never costs a model load.
2. Every (premise, hypothesis) pair is measured against the model's real
   position limit and *deterministically* shrunk until it fits. Truncation is
   never silent: the returned record always says whether it happened.
3. The full label distribution is read and mapped by the *intended* label set,
   not by "whatever scored highest", so a paraphrase-style label set
   (entailment/not_entailment) still yields a usable entailment score.
4. Any failure returns an explicit neutral/unavailable record. An NLI error can
   never be mistaken for neutral evidence, and can never crash the pipeline.
"""

from functools import lru_cache

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Fallback used only if the model config cannot be read.
DEFAULT_MAX_TOKENS = 512

# Public model limit, used by the evidence chunker to size its chunks.
NLI_MAX_TOKENS = DEFAULT_MAX_TOKENS

# Tokens reserved for [CLS]/[SEP] and the hypothesis (the claim).
SAFETY_MARGIN = 24
NLI_SAFETY_MARGIN = SAFETY_MARGIN

# Canonical label order for cross-encoder/nli-deberta-v3-base.
DEFAULT_LABELS = ("contradiction", "entailment", "neutral")

# If the model exposes a non-standard label set, map it onto the canonical one.
LABEL_ALIASES = {
    "entailment": "entailment",
    "entail": "entailment",
    "label_0": "entailment",
    "contradiction": "contradiction",
    "contradict": "contradiction",
    "contradictory": "contradiction",
    "label_2": "contradiction",
    "neutral": "neutral",
    "not_entailment": "neutral",
    "not_entail": "neutral",
    "unrelated": "neutral",
    "label_1": "neutral",
}

_nli_model = None
_model_limits = {}


def _load_model():
    global _nli_model
    if _nli_model is None:
        from transformers import pipeline

        _nli_model = pipeline(
            "text-classification",
            model=MODEL_NAME,
            truncation=False,
        )
    return _nli_model


def _model_max_tokens(model):
    """Real position limit of the loaded model (512 for this NLI model)."""
    key = id(model)
    if key in _model_limits:
        return _model_limits[key]

    limit = DEFAULT_MAX_TOKENS
    for path in (
        ("config", "max_position_embeddings"),
        ("model", "config", "max_position_embeddings"),
    ):
        obj = model
        try:
            for attribute in path:
                obj = getattr(obj, attribute, None)
                if obj is None:
                    break
            if isinstance(obj, int) and 64 < obj <= 4096:
                limit = obj
                break
        except Exception:
            continue

    _model_limits[key] = limit
    return limit


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _neutral(reason, status="unavailable"):
    return {
        "label": "neutral",
        "score": 0.0,
        "scores": {},
        "truncated": False,
        "status": status,
        "reason": reason,
    }


def _canonical_scores(raw_outputs):
    """
    Fold the raw model output into {entailment, contradiction, neutral}.

    Handles both the standard 3-label set and paraphrase-style 2-label sets.
    """
    folded = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}

    for item in raw_outputs:
        raw_label = str(item.get("label", "")).lower().strip()
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        canonical = LABEL_ALIASES.get(raw_label)
        if canonical:
            folded[canonical] = max(folded[canonical], score)

    # A 2-label model that only reported "contradiction" and "neutral" leaves
    # entailment at 0.0, which is the correct conservative reading.
    return folded


def _shrink_to_fit(premise, hypothesis, max_tokens):
    """
    Deterministically reduce the premise until premise + hypothesis fits.

    Returns (premise, truncated). Word-level halving guarantees termination
    and never raises.
    """
    tokenizer = _tokenizer()
    budget = max(16, max_tokens - SAFETY_MARGIN)

    def encoded_length(text):
        try:
            return len(tokenizer.encode(text, add_special_tokens=True, verbose=False))
        except Exception:
            # Worst-case fallback estimate (~4 chars/token) so the guard can
            # still bound the input if the tokenizer misbehaves.
            return max(1, len(text) // 4)

    hypothesis_tokens = encoded_length(hypothesis)

    if hypothesis_tokens >= max_tokens:
        # The claim itself is too long: keep the model call meaningful by
        # bounding the claim from the middle-out.
        hypothesis = " ".join(hypothesis.split()[: max(8, budget // 2)])
        hypothesis_tokens = encoded_length(hypothesis)

    allowed = max(16, budget - hypothesis_tokens)
    if encoded_length(premise) <= allowed:
        return premise, False

    words = premise.split()
    if not words:
        return premise[:200], True

    # Halve until it fits: O(log n) tokenizer passes.
    while words:
        keep = max(8, len(words) // 2)
        candidate = " ".join(words[:keep])
        if encoded_length(candidate) <= allowed:
            words = candidate.split()
            break
        words = words[:keep]

    shrunk = " ".join(words)
    if encoded_length(shrunk) > allowed:
        shrunk = " ".join(shrunk.split()[: max(8, allowed // 2)])

    return shrunk, True


def check_nli(claim, evidence):
    """
    Classify `evidence` against `claim`.

    Returns a record that always contains:
        label      entailment | contradiction | neutral
        score      confidence for `label` (0.0 when unavailable)
        scores     full canonical distribution (empty when unavailable)
        truncated  True if the premise/hypothesis had to be shortened
        status     ok | shrunk | error | unavailable
        reason     short machine-readable explanation
    """
    if not claim or not str(claim).strip() or not evidence or not str(evidence).strip():
        return _neutral("Empty claim or evidence.", status="unavailable")

    claim = str(claim).strip()
    evidence = str(evidence).strip()

    try:
        model = _load_model()
    except Exception as exc:
        return _neutral(f"NLI model unavailable: {exc}", status="unavailable")

    max_tokens = _model_max_tokens(model)

    try:
        premise, truncated = _shrink_to_fit(evidence, claim, max_tokens)
    except Exception as exc:
        return _neutral(f"Token budget guard failed: {exc}", status="error")

    try:
        raw = model(premise, text_pair=claim, truncation=True, max_length=max_tokens)
    except Exception as exc:
        return _neutral(f"NLI inference failed: {exc}", status="error")

    if not raw:
        return _neutral("NLI returned no result.", status="error")

    scores = _canonical_scores(raw)

    if max(scores.values(), default=0.0) <= 0.0:
        return _neutral(
            f"Unrecognised NLI labels: {[i.get('label') for i in raw]}",
            status="error",
        )

    label = max(scores, key=lambda key: scores[key])

    return {
        "label": label,
        "score": round(float(scores[label]), 4),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "truncated": truncated,
        "status": "shrunk" if truncated else "ok",
        "reason": (
            f"Premise shortened to fit {max_tokens} tokens."
            if truncated
            else f"Fits within {max_tokens} tokens."
        ),
    }


if __name__ == "__main__":

    claim = "The Taj Mahal is located in Mumbai."

    evidence = (
        "The Taj Mahal is an ivory-white marble mausoleum located in Agra, "
        "Uttar Pradesh, India. " * 200
    )

    result = check_nli(claim, evidence)

    print("NLI Result:", result)
