import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from glossary import normalize_word


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1].split("_")[-1][:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas] if "test" not in split else list(list())
    answer_labels = (
        [a["labels"] for a in answers] if "test" not in split else list(list())
    )
    answer_scores = (
        [a["scores"] for a in answers] if "test" not in split else list(list())
    )
    answers = (
        [[label2ans[l] for l in al] for al in answer_labels]
        if "test" not in split
        else list(list())
    )

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]


def make_arrow(root, dataset_root, fix_answer_normalization=True):
    with open(f"{root}/vqav2/v2_OpenEnded_mscoco_train2014_questions.json", "r") as fp:
        questions_train2014 = json.load(fp)["questions"]
    with open(f"{root}/vqav2/v2_OpenEnded_mscoco_val2014_questions.json", "r") as fp:
        questions_val2014 = json.load(fp)["questions"]
    with open(f"{root}/vqav2/v2_OpenEnded_mscoco_test2015_questions.json", "r") as fp:
        questions_test2015 = json.load(fp)["questions"]
    with open(f"{root}/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r") as fp:
        questions_test_dev2015 = json.load(fp)["questions"]

    with open(f"{root}/vqav2/v2_mscoco_train2014_annotations.json", "r") as fp:
        annotations_train2014 = json.load(fp)["annotations"]
    with open(f"{root}/vqav2/v2_mscoco_val2014_annotations.json", "r") as fp:
        annotations_val2014 = json.load(fp)["annotations"]

    annotations = dict()

    if fix_answer_normalization:
        suffix_name = '_fix'
    else:
        suffix_name = ''

    for split, questions in zip(
        ["train", "val", "test", "test-dev"],
        [
            questions_train2014,
            questions_val2014,
            questions_test2015,
            questions_test_dev2015,
        ],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(questions):
            _annot[q["image_id"]][q["question_id"]] = [q["question"]]

        annotations[split] = _annot
    
    print(len(annotations['train']), sum([len(i) for i in annotations['train'].values()]))
    print(len(annotations['val']), sum([len(i) for i in annotations['val'].values()]))

    all_major_answers = list()

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        for q in tqdm(annots):
            all_major_answers.append(q["multiple_choice_answer"])

    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            answers = q["answers"]
            answer_count = {}
            for answer in answers:
                if fix_answer_normalization:
                    answer_ = normalize_word(answer["answer"])
                else:
                    answer_ = answer["answer"]
                answer_count[answer_] = answer_count.get(answer_, 0) + 1

            labels = []
            scores = []
            for answer in answer_count:
                if answer not in ans2label:
                    continue
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)

            _annot[q["image_id"]][q["question_id"]].append(
                {"labels": labels, "scores": scores,}
            )
    print(len(annotations['train']), sum([len(i) for i in annotations['train'].values()]))        
    print(len(annotations['val']), sum([len(i) for i in annotations['val'].values()]))
    # #image #question
    # 82783 443757
    # 40504 214354

    for split in ["train", "val"]:
        filtered_annot = dict()
        for ik, iv in annotations[split].items():
            new_q = dict()
            for qk, qv in iv.items():
                if len(qv[1]["labels"]) != 0:
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot
    
    print(len(annotations['train']), sum([len(i) for i in annotations['train'].values()]), sum([len(qa) for q in annotations['train'].values() for qa in q.values()]))
    print(len(annotations['val']), sum([len(i) for i in annotations['val'].values()]), sum([len(qa) for q in annotations['val'].values() for qa in q.values()]))
    #       82774 434867 869734
    #       40503 210051 420102
    # fix:  82774 435174 870348
    # fix:  40503 210207 420414

    for split in [
        "train",
        "val",
        "test",
        "test-dev",
    ]:
        annot = annotations[split]
        split_name = {
            "train": "train2014",
            "val": "val2014",
            "test": "test2015",
            "test-dev": "test2015",
        }[split]
        paths = list(glob(f"{root}/{split_name}/*.jpg"))
        random.shuffle(paths)
        annot_paths = [
            path
            for path in paths
            if int(path.split("/")[-1].split("_")[-1][:-4]) in annot
        ]

        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(annot_paths), len(annot),
        )

        bs = [
            path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/vqav2_{split}{suffix_name}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(f"{dataset_root}/vqav2_val{suffix_name}.arrow", "r")
    ).read_all()

    pdtable = table.to_pandas()

    df1 = pdtable[:-1000]
    df2 = pdtable[-1000:]

    df1 = pa.Table.from_pandas(df1)
    df2 = pa.Table.from_pandas(df2)

    with pa.OSFile(f"{dataset_root}/vqav2_trainable_val{suffix_name}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, df1.schema) as writer:
            writer.write_table(df1)

    with pa.OSFile(f"{dataset_root}/vqav2_rest_val{suffix_name}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, df2.schema) as writer:
            writer.write_table(df2)

make_arrow('~/BT/dataset/mscoco_flickr30k_vqav2_snli_ve', '~/BT/dataset/fine-tune', True)
make_arrow('~/BT/dataset/mscoco_flickr30k_vqav2_snli_ve', '~/BT/dataset/fine-tune', False)