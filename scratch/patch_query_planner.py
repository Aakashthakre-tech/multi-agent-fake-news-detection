from pathlib import Path

path = Path("query_planner.py")
content = path.read_text(encoding="utf-8")

old_block = '''    for phrase in phrases[:2]:
        if not phrase:
            continue
        if all(token in covered for token in phrase.lower().split()):
            continue
        parts.append(_quote(phrase))
        covered.update(phrase.lower().split())'''

new_block = '''    for phrase in phrases[:2]:
        if not phrase:
            continue
        phrase_tokens = set(phrase.lower().split())
        if covered and len(phrase_tokens & covered) >= len(phrase_tokens) * 0.5:
            continue
        if all(token in covered for token in phrase.lower().split()):
            continue
        parts.append(_quote(phrase))
        covered.update(phrase.lower().split())'''

assert old_block in content, "old_block not found in query_planner.py"
content = content.replace(old_block, new_block)

old_date_block = '''    if primary_date and variants:
        if location_quoted and event_anchor:
            add(
                "date_loc_keywords",
                "date",
                _join(_quote(variants[0]), location_quoted, event_anchor),
            )
        if location_quoted and phrase_quoted:'''

new_date_block = '''    if primary_date and variants:
        if location_quoted and event_anchor:
            add(
                "date_loc_keywords",
                "date",
                _join(_quote(variants[0]), location_quoted, event_anchor),
            )
        elif event_anchor:
            add(
                "date_loc_keywords",
                "date",
                _join(_quote(variants[0]), event_anchor),
            )
        if location_quoted and phrase_quoted:'''

assert old_date_block in content, "old_date_block not found in query_planner.py"
content = content.replace(old_date_block, new_date_block)

path.write_text(content, encoding="utf-8")
print("Successfully patched query_planner.py")
