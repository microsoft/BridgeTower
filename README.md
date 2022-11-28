# BridgeTower

This repo is the official `Pytorch` implementation of ["BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning"](https://arxiv.org/abs/2206.08657).

## Updates

- Nov, 2022: BridgeTower got accepted by [AAAI'23](https://aaai.org/Conferences/AAAI-23/). Code and checkpoints are released.
- Jun, 2022: We release the preprint version in [Arxiv](https://arxiv.org/abs/2206.08657).
- May, 2022: BridgeTower (single model, 4M data) achieved 78.73% and 81.15% (base and large) on the [VQAv2 Challenge](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278) test-std set.

## Abstract

Vision-Language (VL) models with the Two-Tower architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a deep cross-modal encoder, or feed the last-layer uni-modal representations from the deep pre-trained uni-modal encoders into the top cross-modal encoder. Both approaches potentially restrict vision-language representation learning and limit model performance. In this paper, we propose BridgeTower, which introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BridgeTower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets.

## Architecture

![Architecture](images/framework.jpg)

## Main Results

![Result1](images/result1.jpg)

![Result2](images/result2.jpg)

## Deployment

- Run `setup.sh` to set up the environment.
- [Optional] We use [wandb](https://wandb.ai/) to track experiments! Please remember to `wandb login` and paste your token before running the script.

## Dataset Preparation

- We follow [ViLT](https://github.com/dandelin/ViLT) and use pyarrow to serialize the datasets. See [here](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.
- For SNLI-VE dataset, we follow [here](https://github.com/necla-ml/SNLI-VE).
- For VG-QA dataset, except the image-text pairs in [VG](https://visualgenome.org/api/v0/api_home.html) got from [here](https://github.com/dandelin/ViLT/blob/master/DATA.md), [image meta data](https://visualgenome.org/static/data/dataset/image_data_v1.json.zip), [question answers data](https://visualgenome.org/static/data/dataset/question_answers.json.zip) and [coco split information](https://github.com/peteanderson80/bottom-up-attention/tree/master/data/genome/coco_splits) also need to be downloaded.
- The final file structure of datasets are shown in `setup.sh`.

## Checkpoints

- Pre-trained checkpoints on 4M data: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_pt_base.ckpt?sv=2020-10-02&st=2022-11-24T12%3A18%3A49Z&se=2027-11-25T12%3A18%3A00Z&sr=b&sp=r&sig=BJigddAMHfNUtQuTGH8bJUrzAO3LfaeSm48AXUqZngY%3D) and [LARGE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_pt_large.ckpt?sv=2020-10-02&st=2022-11-24T12%3A19%3A19Z&se=2027-11-25T12%3A19%3A00Z&sr=b&sp=r&sig=8yWqesQACrJSi0JMLIA0uAbNlMQKb653gOXjXjQuIW4%3D)
- Fine-tuned checkpoints for
  - Visual Question Answering on VQAv2: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_vqav2.ckpt?sv=2020-10-02&st=2022-11-24T12%3A16%3A38Z&se=2027-11-25T12%3A16%3A00Z&sr=b&sp=r&sig=t35v4kezDcSOm9Q9E767PhNGAQRsiYm%2FMSDgHIz%2Fvto%3D), [BASE(w/ VGQA)](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_vqav2_vgqa.ckpt?sv=2020-10-02&st=2022-11-24T12%3A17%3A18Z&se=2027-11-25T12%3A17%3A00Z&sr=b&sp=r&sig=BD%2BOsI%2F6R905vBJUlrWlgx3%2BmaBRsa2rQcHBChhW0eE%3D), [LARGE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_large_vqav2.ckpt?sv=2020-10-02&st=2022-11-24T12%3A17%3A47Z&se=2027-11-25T12%3A17%3A00Z&sr=b&sp=r&sig=RqL7Eeye4385oaO1nvVvRwC4d%2ByhpEVGM3xmS4GcKkQ%3D), [LARGE(w/ VGQA)](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_large_vqav2_vgqa.ckpt?sv=2020-10-02&st=2022-11-24T12%3A18%3A29Z&se=2027-11-25T12%3A18%3A00Z&sr=b&sp=r&sig=xtI8rmEqjMmN1b1bcE0KB9ePUax3SuRfOt%2Bp2ATH9ng%3D)
  - Image-Text Retrieval on Flickr30k: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_irtr_itm_itc_f30k.ckpt?sv=2020-10-02&st=2022-11-24T12%3A13%3A42Z&se=2027-11-25T12%3A13%3A00Z&sr=b&sp=r&sig=0BP3pOiE4AFkK4BTgQl5Dy6iJWxHuJffpjU4LFMTfWY%3D)
  - Visual Entailment on SNLI-VE: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_snlive.ckpt?sv=2020-10-02&st=2022-11-24T12%3A15%3A27Z&se=2027-11-25T12%3A15%3A00Z&sr=b&sp=r&sig=IccPmnxQYIpWO8m6kwtEFir9wmVq1SsLOqmw0FRc9hY%3D)
  - Visual Reasoning on NLVR$^2$: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_nlvr2.ckpt?sv=2020-10-02&st=2022-11-24T12%3A15%3A09Z&se=2027-11-25T12%3A15%3A00Z&sr=b&sp=r&sig=AL3q15eyhPBHaWY0FOop9goHVq8CbNluABDk%2FS94rkI%3D)
  - Image-Text Retrieval on MSCOCO: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_irtr_itm_itc_coco.ckpt?sv=2020-10-02&st=2022-11-24T12%3A13%3A18Z&se=2027-11-25T12%3A13%3A00Z&sr=b&sp=r&sig=ahM%2FyI8fg9D4obCZsNKaxLzPVz2y8RX8ydZNToGavC4%3D)

## Pre-training on Image-Text Datasets

```bash
# Pre-train BridgeTower Base Model
bash scripts/pre_train.sh
# Pre-train BridgeTower Large Model
bash scripts/pre_train_large.sh
```

## Fine-tuning on Downstream VL Tasks

- VQAv2 Evaluation needs to submit the `json` file in the `logs/` directory to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.

```bash
# Base Model on VQAv2 without VLP
bash scripts/ftfs_base_vqa.sh

# Large Model on VQAv2 without VLP
bash scripts/ftfs_large_vqa.sh

# Base Model on VQAv2 with VLP
bash scripts/ftfpt_base_vqa.sh

# Large Model on VQAv2 with VLP
bash scripts/ftfpt_large_vqa.sh

# Base Model on IRTR-Flickr30K with VLP (directly use ITM with multiple false texts)
bash scripts/ftfpt_base_irtr_f30k.sh

# Base Model on IRTR-Flickr30K with VLP (follow ALBEF to use ITC to sample hard negatives for ITM)
bash scripts/ftfpt_base_irtr_itm_itc_f30k.sh

# Base Model on SNLI-VE with VLP
bash scripts/ftfpt_base_snlive.sh

# Base Model on NLVR^2 with VLP
bash scripts/ftfpt_base_nlvr2.sh

# Base Model on IRTR-MSCOCO with VLP (follow ALBEF to use ITC to sample hard negatives for ITM)
bash scripts/ftfpt_base_irtr_itm_itc_coco.sh

```

## Fine-tuning on Uni-Modal Tasks

```bash
# Base Model on CIFAR with VLP
bash scripts/ftfpt_base_cifar.sh

# Base Model on GLUE with VLP
bash scripts/ftfpt_base_glue.sh
```

## Citation

```
@article{xu2022bridge,
  title={BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning},
  author={Xu, Xiao and Wu, Chenfei and Rosenman, Shachar and Lal, Vasudev and Che, Wanxiang and Duan, Nan},
  journal={arXiv preprint arXiv:2206.08657},
  year={2022}
}
```

## Acknowledgement

We are highly grateful for the public code of the following papers, our code is partly based on them:

- Main Code: [METER](https://github.com/zdou0830/METER), [ViLT](https://github.com/dandelin/ViLT)
- Others: [CLIP](https://github.com/openai/CLIP), [ALBEF](https://github.com/salesforce/ALBEF), [BLIP](https://github.com/salesforce/BLIP)
