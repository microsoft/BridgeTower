import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob
import pandas as pd

def path2rest(path, iid2captions):
    split, _, name = path.split("/")[-3:]
    split = split.split("_")[-1]
    iid = name

    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]

    return [
        binary,
        captions,
        iid,
        split,
    ]

annotations = {
    'val' : 'Validation_GCC-1.1.0-Validation.tsv',
    'train' : 'Train_GCC-training.tsv'
}

check_exist = {
    'val' : 'val_image_exist.txt',
    'train' : 'train_image_exist.txt'
}

def make_arrow(root, dataset_root):
    for split in ["val", "train"]:
        data = pd.read_csv(f"{root}/utils/{annotations[split]}", sep='\t', header=None)
        with open(f"{root}/{check_exist[split]}", 'r') as fr:
            data_exist = fr.readlines()
        exist_image_file_names = [line.strip().split("/")[-1] for line in data_exist]
        iid2captions = dict()
        captions = [dataitem[0] for dataitem in data.values.tolist()]
        for exist_image_file_name in tqdm(exist_image_file_names):
            exist_image_idx = int(exist_image_file_name.split(".")[0])
            iid2captions[exist_image_file_name] = [captions[exist_image_idx]]

        paths = list(glob(f"{root}/{split}_image/*"))
        random.shuffle(paths)
        caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
        if len(paths) == len(caption_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(caption_paths), len(iid2captions),
        )

        sub_len = int(len(caption_paths) // 100000)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = caption_paths[sub * 100000 : (sub + 1) * 100000]
            bs = [path2rest(path, iid2captions) for path in tqdm(sub_paths)]
            dataframe = pd.DataFrame(
                bs, columns=["image", "caption", "image_id", "split"],
            )

            table = pa.Table.from_pandas(dataframe)

            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/conceptual_caption_{split}_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del bs
            gc.collect()

make_arrow('~/BT/dataset/cc', '~/BT/dataset/pre-train')