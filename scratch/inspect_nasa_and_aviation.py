import sys
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from tools import scrape_url
from evidence_analyzer import split_into_chunks, check_nli, check_date_compatibility, extract_claim_attributes

claim = "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969."
attrs = extract_claim_attributes(claim)

for url in [
    "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind",
    "https://www.thisdayinaviation.com/20-july-1969-1056-p-m-edt",
]:
    print("=" * 80)
    print("URL:", url)
    text = scrape_url.invoke(url)
    chunks = split_into_chunks(text)
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        if any(w in chunk.lower() for w in ["first", "moon", "landed", "armstrong"]):
            nli = check_nli(claim, chunk)
            dc = check_date_compatibility(claim, chunk, attrs)
            print(f"Chunk {i} [date={dc['status']} nli={nli['label']} {nli['score']:.4f}]:")
            print(chunk[:180].replace("\n", " "))
            print()
