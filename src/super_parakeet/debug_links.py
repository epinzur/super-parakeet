import json
from tqdm import tqdm

from typing import Dict, List

with open("debug_links.json", "r") as f:
    links = json.load(f)
    in_links: List[str] = links["in_links"]
    out_links: List[str] = links["out_links"]
    bidir_links: Dict[str, int] = links["bidir_links"]

print(f"Links In: {len(in_links)}, Out: {len(out_links)}, BiDir: {len(bidir_links)}")

matches = []
for out_link in tqdm(out_links):
    if out_link in in_links:
        matches.append(out_link)

for bidir_link, count in bidir_links.items():
    if count > 0:
        matches.append(bidir_link)

with open("matching_links.txt", "w") as f:
    for link in sorted(matches):
        f.write(f"{link}\n")

print(f"Found {len(matches)} matching links")
