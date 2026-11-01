import sys
sys.path.insert(0, '.')

with open('pipeline.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

old = '''        content = scrape_url.invoke({"url": url})

        if (
            not content
            or content.startswith("SCRAPE_FAILED:")
            or content.startswith("Could not scrape URL:")
        ):
            fallback_url = _scientific_fallback_url(url)

            if fallback_url:
                fallback_content = scrape_url.invoke({"url": fallback_url})

            if (
                fallback_url
                and fallback_content
                and not fallback_content.startswith("SCRAPE_FAILED:")
                and not fallback_content.startswith("Could not scrape URL:")
                and len(fallback_content.strip()) >= 180
            ):
                url = fallback_url
                domain = extract_domain(url)
                title = "FAQs About Water and Steam"
                content = fallback_content


        # Determine scrape success
        is_scrape_success = bool(
            content
            and not content.startswith("SCRAPE_FAILED:")
            and not content.startswith("Could not scrape URL:")
            and len(content.strip()) >= 180
        )'''

new = '''        scrape_result = scrape_url_detailed(url)
        content = scrape_result["content"]
        scrape_diag = {
            "url": url,
            "domain": domain,
            "title": title,
            "scrape_status": scrape_result["scrape_status"],
            "extraction_method": scrape_result["extraction_method"],
            "char_count": scrape_result["char_count"],
            "http_status": scrape_result["http_status"],
            "retry_count": scrape_result["retry_count"],
            "error_category": scrape_result["error_category"],
        }

        if (
            not content
            or scrape_result["scrape_status"] == "failed"
        ):
            fallback_url = _scientific_fallback_url(url)

            if fallback_url:
                fallback_result = scrape_url_detailed(fallback_url)
                fallback_content = fallback_result["content"]

                if (
                    fallback_url
                    and fallback_content
                    and fallback_result["scrape_status"] == "success"
                    and len(fallback_content.strip()) >= 180
                ):
                    url = fallback_url
                    domain = extract_domain(url)
                    title = "FAQs About Water and Steam"
                    content = fallback_content
                    scrape_diag = {
                        "url": url,
                        "domain": domain,
                        "title": title,
                        "scrape_status": fallback_result["scrape_status"],
                        "extraction_method": fallback_result["extraction_method"],
                        "char_count": fallback_result["char_count"],
                        "http_status": fallback_result["http_status"],
                        "retry_count": fallback_result["retry_count"],
                        "error_category": fallback_result["error_category"],
                    }


        # Determine scrape success
        is_scrape_success = bool(
            content
            and scrape_result["scrape_status"] == "success"
            and len(content.strip()) >= 180
        )'''

if old in content:
    content = content.replace(old, new, 1)
    with open('pipeline.py', 'w', encoding='utf-8', errors='replace') as f:
        f.write(content)
    print("Replacement 1 successful")
else:
    print("OLD STRING NOT FOUND")
    lines = content.split('\n')
    for i, line in enumerate(lines[418:422], start=419):
        print(f'{i}: {repr(line)}')