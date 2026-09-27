from evidence_matcher import calculate_similarity

claim = "The Taj Mahal is located in Mumbai."

evidence = """
The Taj Mahal is an ivory-white marble mausoleum
located in Agra, Uttar Pradesh, India.
"""

score = calculate_similarity(claim, evidence)

print("Similarity:", score)