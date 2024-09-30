
import json

from typing import Set
from tqdm import tqdm

dev_snippets_path = "datasets/complexwebquestions_V1_1/web_snippets_dev.json"
test_snippets_path = "datasets/complexwebquestions_V1_1/web_snippets_test.json"

in_snippets_paths = [
    dev_snippets_path,
    test_snippets_path,
]

out_snippets_path = "datasets/complexwebquestions_V1_1/web_snippets.jsonl"
out_snippets = open(out_snippets_path, "w")
all_titles: Set[str] = set()


for in_snippets_path in tqdm(in_snippets_paths):
    with open(in_snippets_path, "r") as in_file:
        data = json.load(in_file)

    for qs_set in tqdm(data):
        lines = []

        for web_snippet in qs_set["web_snippets"]:
            title = web_snippet["title"]
            if title in all_titles:
                continue
            else:
                all_titles.add(title)
                lines.append(json.dumps({"title": title, "snippet": web_snippet["snippet"]}) + "\n")

        out_snippets.writelines(lines)

out_snippets.close()
