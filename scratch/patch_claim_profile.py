from pathlib import Path

path = Path("claim_profile.py")
content = path.read_text(encoding="utf-8")

old_kw = '''def extract_keywords(text, entity_words=None):
    """Content words of length >= 3 that are not stopwords or entity words."""
    if not text:
        return []

    entity_words = entity_words or set()

    words = re.findall(r"\\b[A-Za-z][A-Za-z'-]{2,}\\b", text)

    keywords = []
    for word in words:
        lower = word.lower()
        if lower in STOPWORDS or lower in entity_words:
            continue
        if lower not in keywords:
            keywords.append(lower)

    return keywords


def entity_word_set(entities):
    words = set()
    for entity in entities:
        for token in re.findall(r"[A-Za-z]{2,}", entity):
            words.add(token.lower())
    return words


def extract_phrases(text, keywords, max_phrases=6):
    """
    Multi-word content phrases ('commercial building', 'heavy rains').

    Built from adjacent non-stopword tokens so they read as event phrases
    rather than bag-of-words noise.
    """
    tokens = re.findall(r"\\b[A-Za-z][A-Za-z'-]*\\b", text)'''

new_kw = '''def extract_keywords(text, entity_words=None):
    """Content words that are not stopwords or entity words (including alphanumeric terms like '11', 'covid-19')."""
    if not text:
        return []

    entity_words = entity_words or set()

    words = re.findall(r"\\b[A-Za-z0-9][A-Za-z0-9'-]*\\b", text)

    keywords = []
    for word in words:
        lower = word.lower()
        if len(lower) < 2 or (len(lower) < 3 and lower.isalpha()):
            continue
        if lower in STOPWORDS or lower in entity_words:
            continue
        if lower not in keywords:
            keywords.append(lower)

    return keywords


def entity_word_set(entities):
    words = set()
    for entity in entities:
        for token in re.findall(r"[A-Za-z0-9]{2,}", entity):
            words.add(token.lower())
    return words


def extract_phrases(text, keywords, max_phrases=6):
    """
    Multi-word content phrases ('commercial building', 'heavy rains', 'apollo 11 mission').

    Built from adjacent non-stopword tokens so they read as event phrases
    rather than bag-of-words noise.
    """
    tokens = re.findall(r"\\b[A-Za-z0-9][A-Za-z0-9'-]*\\b", text)'''

assert old_kw in content, "old_kw not found in claim_profile.py"
content = content.replace(old_kw, new_kw)
path.write_text(content, encoding="utf-8")
print("Successfully patched claim_profile.py")
