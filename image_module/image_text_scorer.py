def calculate_consistency_score(clip_similarity):
    """
    Convert CLIP similarity into a preliminary
    normalized image-text consistency score.

    This is NOT a probability of truth.
    """

    clip_normalized = (clip_similarity + 1) / 2

    clip_normalized = max(
        0.0,
        min(1.0, clip_normalized)
    )

    return round(clip_normalized * 100, 2)