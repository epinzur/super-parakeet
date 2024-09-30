
from pathlib import Path
import json

dev_path = "datasets/complexwebquestions_V1_1/ComplexWebQuestions_dev.json"
test_path = "datasets/complexwebquestions_V1_1/ComplexWebQuestions_test.json"
train_path = "datasets/complexwebquestions_V1_1/ComplexWebQuestions_train.json"

in_paths = [
    dev_path,
    test_path,
    train_path,
]

keys_to_drop = ["composition_answer", "created", "machine_question", "sparql", "webqsp_ID", "webqsp_question"]

for in_file_path in in_paths:
    in_stem = Path(in_file_path).stem
    in_path = Path(in_file_path).parent

    out_file_path = Path.joinpath(in_path, f"{in_stem}_qa.jsonl")

    with open(in_file_path, "r") as in_file:
        data = json.load(in_file)

    out_file = open(out_file_path, "w")

    for qa_set in data:
        if "answers" in qa_set:
            answers = qa_set["answers"]
            if len(answers) != 1: # skip questions with multiple answers
                continue
            qa_set["answer"] = answers[0]["answer"]
            qa_set["aliases"] = answers[0]["aliases"]
            del qa_set["answers"]

        for key in keys_to_drop:
            if key in qa_set:
                del qa_set[key]

        out_file.write(json.dumps(qa_set) + "\n")

    out_file.close()
