
from pathlib import Path
import json

from tqdm import tqdm
from typing import Set

dev_distractor_v1_path = "datasets/hotpotqa/hotpot_dev_distractor_v1.json"
dev_fullwiki_v1_path = "datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
test_fullwiki_v1_path = "datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
train_v1_1_path = "datasets/hotpotqa/hotpot_train_v1.1.json"

in_paths = [
    dev_distractor_v1_path,
    dev_fullwiki_v1_path,
    test_fullwiki_v1_path,
    train_v1_1_path,
]

facts_out_path = "datasets/hotpotqa/facts.jsonl"

facts_out = open(facts_out_path, "w")


facts: Set[str] = set()

for in_file_path in tqdm(in_paths):
    in_stem = Path(in_file_path).stem
    in_path = Path(in_file_path).parent

    out_file_path = Path.joinpath(in_path, f"{in_stem}_qa.jsonl")

    with open(in_file_path, "r") as in_file:
        data = json.load(in_file)

    out_file = open(out_file_path, "w")

    for qa_set in tqdm(data):
        if "context" in qa_set:
            context = qa_set["context"]
            del qa_set["context"]

            for paragraph in context:
                fact = paragraph[0]
                if fact in facts:
                    continue
                else:
                    facts.add(fact)
                    text = "".join(paragraph[1])
                    facts_out.write(json.dumps({"fact": fact, "text": text}) + "\n")

        if "supporting_facts" in qa_set:
            del qa_set["supporting_facts"]

        out_file.write(json.dumps(qa_set) + "\n")


    out_file.close()

facts_out.close()
