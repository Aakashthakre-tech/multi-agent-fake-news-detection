"""
evidence_matcher.py
===================
Semantic similarity between a claim and an evidence chunk.

Similarity is a retrieval/pre-filter signal only. It is never allowed to
classify evidence on its own (see evidence_analyzer.classify_evidence).

The claim embedding is cached and evidence chunks are encoded in batches, which
removes the per-chunk re-encoding of the claim that the original version did.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunks are encoded in batches; very small batches are slower than one pass.
BATCH_SIZE = 32


@lru_cache(maxsize=8)
def _encode_claim(claim: str):
    return model.encode(claim, convert_to_tensor=True, show_progress_bar=False)


def calculate_similarity(claim: str, evidence: str) -> float:
    """Cosine similarity for a single evidence string."""
    claim_embedding = _encode_claim(claim)

    evidence_embedding = model.encode(
        evidence,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    similarity = util.cos_sim(
        claim_embedding,
        evidence_embedding
    ).item()

    return round(float(similarity), 4)


def calculate_similarity_batch(claim: str, evidences) -> list:
    """
    Cosine similarity for many evidence strings against one claim.

    Returns a list of rounded floats aligned with `evidences`.
    """
    evidences = list(evidences)

    if not evidences:
        return []

    claim_embedding = _encode_claim(claim)

    evidence_embeddings = model.encode(
        evidences,
        convert_to_tensor=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )

    similarities = util.cos_sim(
        claim_embedding,
        evidence_embeddings
    )

    return [round(float(value), 4) for value in similarities.flatten().tolist()]
