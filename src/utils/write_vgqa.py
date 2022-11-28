import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from glossary import normalize_word


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1][:-4])

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


def make_arrow(root, vg_root, dataset_root, use_coco_images_only=False):
    with open(f"{root}/vqav2/v2_mscoco_train2014_annotations.json", "r") as fp:
        annotations_train2014 = json.load(fp)["annotations"]
    with open(f"{root}/vqav2/v2_mscoco_val2014_annotations.json", "r") as fp:
        annotations_val2014 = json.load(fp)["annotations"]

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

    if use_coco_images_only:
        id_file_name = 'vgqa/coco_ids.json'
        output_file_name = 'vgqa_coco'
    else:
        id_file_name = 'vgqa/ids.json'
        output_file_name = 'vgqa'
    with open(os.path.join(vg_root, id_file_name)) as f:
        ids = json.load(f)
        train_image_ids = ids['train']
        val_image_ids = ids['val'] + ids['test']

    with open(f"{vg_root}/annotations/question_answers.json", "r") as fp:
        qa_annotations = json.load(fp)
    
    annotations = dict()
    annotations['train'] = defaultdict(dict)
    annotations['val'] = defaultdict(dict)
    qa_images, qa_valid_images, qa_pairs, qa_valid_pairs = 0, 0, 0, 0
    scores_counter = []
    for annots in tqdm(qa_annotations):
        qas = annots['qas']
        split = None
        if len(qas) == 0:
            continue
        if qas[0]['image_id'] in train_image_ids:
            split = 'train'
        elif qas[0]['image_id'] in val_image_ids:
            split = 'val'
        if split is not None:
            qa_images += 1
            qa_pairs += len(qas)
            qa_valid_image_flag = 0
            question_answer = defaultdict(dict)
            for qa in qas:
                answer = normalize_word(qa['answer'])
                question = qa['question']
                if answer in ans2label.keys():
                    question_answer[question][answer] = question_answer[question].get(answer, 0) + 1
            # calculate distribution of question_answers
            for q in question_answer:
                # if any(count != 1 for count in question_answer[q].values()):
                scores = sorted(list(question_answer[q].values()))
                scores_counter.append(str(scores))
            # only choose the question with only the same answer (99% of the samples)
            for qa in qas:
                answer = normalize_word(qa['answer'])
                question = qa['question']
                if answer in ans2label.keys() and len(question_answer[question]) == 1:
                    annotations[split][qa['image_id']][qa['qa_id']] = [question, {"labels": [ans2label[answer]], "scores": [1.0],}]
                    question_answer[question] = []
                    qa_valid_image_flag = 1
                    qa_valid_pairs += 1
            qa_valid_images += qa_valid_image_flag

    print(f"qa_images: {qa_images}, qa_valid_images: {qa_valid_images}, qa_pairs: {qa_pairs}, qa_valid_pairs: {qa_valid_pairs}, (train: {sum([len(image) for image in annotations['train'].values()])}, val: {sum([len(image) for image in annotations['val'].values()])})")
    # coco_ids      49663 48645 727063 491809 (450987, 40822) (the same with mcan-vqa, BUTD use 485000 qa pairs)
    # coco_ids_rq   49663 48640 727063 467916 (429061, 38855) (remove question with multiple answers)
    # ids           99280 97217 1445322 978121 (887964, 90157)
    # ids_rq        99280 97207 1445322 931866 (846015, 85851) (remove question with multiple answers)

    distribution = {k: v for k, v in Counter(scores_counter).items()}
    distribution = sorted(distribution.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    print(distribution)
    print(sum([a[1] for a in distribution if len(eval(a[0])) != 1]))
    # 4463 in 472379 / 8576 in 940442

    id2filepath = {}
    with open(os.path.join(vg_root, 'image_data.json')) as f:
        metadata = json.load(f)
        for item in metadata:
            directory = item['url'].split("/")[-2]
            name = item['url'].split("/")[-1]
            filepath = f"{directory}/{name}"
            id2filepath[item['image_id']] = filepath

    for split in [
        "train",
        "val",
    ]:
        annot = annotations[split]
        paths = list(glob(f"{vg_root}/images/VG_100K/*.jpg")) + list(
            glob(f"{vg_root}/images/VG_100K_2/*.jpg")
        )
        random.shuffle(paths)
        annot_paths = [
            path
            for path in paths
            if int(path.split("/")[-1][:-4]) in annot
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
        with pa.OSFile(f"{dataset_root}/{output_file_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

make_arrow('~/BT/dataset/mscoco_flickr30k_vqav2_snli_ve', '~/BT/dataset/vg', '~/BT/dataset/fine-tune', True)
make_arrow('~/BT/dataset/mscoco_flickr30k_vqav2_snli_ve', '~/BT/dataset/vg', '~/BT/dataset/fine-tune', False)