#!/bin/bash
# General
## root dir, you can change it to your specified dir, but remember to change the path in the following script, root_dir in the config.py and src/utils/write_xxx.py
sudo mkdir -p ~/BT
sudo chmod -R 777 ~/BT
cd ~/BT

## plugins
sudo apt-get install software-properties-common tmux net-tools

# BridgeTower
## update Python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv

## create virtual environment
cd ~/BT
python3.8 -m venv venv_BridgeTower
# echo "alias venv_BridgeTower='source ~/BT/venv_BridgeTower/bin/activate'" >> ~/.bashrc
# echo "source ~/BT/venv_BridgeTower/bin/activate" >> ~/.bashrc
source ~/.bashrc

## git clone
cd ~/BT
mkdir BridgeTower
# Please put here our code
cd ~/BT/BridgeTower

## dependency
source ~/BT/venv_BridgeTower/bin/activate
pip install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
pip install evalai
pip install --upgrade requests click python-dateutil

## mkdir
sudo mkdir -p ~/BT
sudo mkdir -p ~/BT/dataset/
sudo mkdir -p ~/BT/best_checkpoints/
sudo mkdir -p ~/BT/checkpoints/
sudo mkdir -p ~/BT/logs/
sudo chmod -R 777 ~/BT

## download data and checkpoints, and put them in ~/BT/dataset/ and ~/BT/best_checkpoints/
## the final file structure of ~/BT/dataset/ should be like this:
# root
#  └── dataset
#      ├── pre-train
#      ├── fine-tune
#      ├── sbu
#      ├── cc
#      ├── nlvr
#      │   ├── dev
#      │   ├── images
#      │   ├── nlvr
#      │   ├── nlvr2
#      │   ├── test1
#      │   └── README.md
#      ├── vg
#      │   ├── annotations
#      │   ├── coco_splits
#      │   ├── images
#      │   ├── vgqa
#      │   └── image_data.json
#      └── mscoco_flickr30k_vqav2_snli_ve
#          ├── flickr30k-images
#          ├── karpathy
#          ├── snli_ve
#          ├── test2015
#          ├── train2014
#          ├── val2014
#          └── vqav2

## then run the src/utils/write_xxx.py to convert the dataset to pyarrow binary file.

# python src/utils/write_coco_karpathy.py
# python src/utils/write_conceptual_caption.py
# python src/utils/write_f30k_karpathy.py
# python src/utils/write_nlvr2.py
# python src/utils/write_sbu.py
# python src/utils/write_vg.py
# python src/utils/write_vqa.py
# python src/utils/vgqa_split.py
# python src/utils/write_vgqa.py
# cp ~/BT/dataset/pre-train/coco_caption_karpathy_* ~/BT/dataset/fine-tune