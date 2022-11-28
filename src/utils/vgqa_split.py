# Based on https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/create_splits.py
''' Determine visual genome data splits to avoid contamination of COCO splits.'''

import argparse
import os
import random
from random import shuffle
import shutil
import subprocess
import sys
import json

random.seed(0) # Make dataset splits repeatable

# The root directory which holds all information of the dataset.
dataDir = '~/BT/dataset/vg'

# First determine train, val, test splits (x, 5000, 5000)
train = set()
val = set()
test = set()

# Load coco test ids
coco_test_ids = set()
with open(os.path.join(dataDir, 'coco_splits/image_info_test2014.json')) as f:
    coco_data = json.load(f)
    for item in coco_data['images']:
        coco_test_ids.add(item['id'])
print("There are %d coco test images" % len(coco_test_ids))

# Load karpathy coco splits
karpathy_train = set()
with open(os.path.join(dataDir, 'coco_splits/karpathy_train_images.txt')) as f:
    for line in f.readlines():
        image_id=int(line.split('.')[0].split('_')[-1])
        karpathy_train.add(image_id)
    
karpathy_val = set()
with open(os.path.join(dataDir, 'coco_splits/karpathy_val_images.txt')) as f:
    for line in f.readlines():
        image_id=int(line.split('.')[0].split('_')[-1])
        karpathy_val.add(image_id)
    
karpathy_test = set()
with open(os.path.join(dataDir, 'coco_splits/karpathy_test_images.txt')) as f:
    for line in f.readlines():
        image_id=int(line.split('.')[0].split('_')[-1])
        karpathy_test.add(image_id)
print( "Karpathy splits are %d, %d, %d (train, val, test)" % (len(karpathy_train), len(karpathy_val), len(karpathy_test)))
    
    
# Load VG image metadata
coco_ids = set()
with open(os.path.join(dataDir, 'image_data.json')) as f:
    metadata = json.load(f)
    for item in metadata:
        if item['coco_id']:
            coco_ids.add(item['coco_id'])

print( "Found %d visual genome images claiming coco ids" % len(coco_ids))
print( "Overlap with COCO test is %d" % len(coco_test_ids & coco_ids))
print( "Overlap with Karpathy train is %d" % len(karpathy_train & coco_ids))
print( "Overlap with Karpathy val is %d" % len(karpathy_val & coco_ids))
print( "Overlap with Karpathy test is %d" % len(karpathy_test & coco_ids))
# Output
# There are 40775 coco test images
# Karpathy splits are 113287, 5000, 5000 (train, val, test)
# Found 51208 visual genome images claiming coco ids
# Overlap with COCO test is 0
# Overlap with Karpathy train is 46944
# Overlap with Karpathy val is 2126
# Overlap with Karpathy test is 2138


# Determine splits
remainder = []
for item in metadata:
    if item['coco_id']:
        if item['coco_id'] in karpathy_train:
            train.add(item['image_id'])
        elif item['coco_id'] in karpathy_val:
            val.add(item['image_id'])
        elif item['coco_id'] in karpathy_test:
            test.add(item['image_id'])    
        else:
            remainder.append(item['image_id'])
    else:
        remainder.append(item['image_id'])

with open(os.path.join(dataDir, 'vgqa/coco_ids.json'), 'w') as f:
    json.dump({'train': list(train), 'val': list(val), 'test': list(test)}, f)

shuffle(remainder)
while len(test) < 5000:
    test.add(remainder.pop())
while len(val) < 5000:
    val.add(remainder.pop())
train |= set(remainder)

assert len(test) == 5000
assert len(val) == 5000
assert len(train) == len(metadata) - 10000
print(len(train)) # 98077
with open(os.path.join(dataDir, 'vgqa/ids.json'), 'w') as f:
    json.dump({'train': list(train), 'val': list(val), 'test': list(test)}, f)

