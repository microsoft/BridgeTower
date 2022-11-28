# Based on https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html

import copy
import os
from datetime import datetime
from typing import Optional
from pytorch_lightning.loggers import WandbLogger
import datasets
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torchvision.datasets import CIFAR10, CIFAR100
from src.modules.clip_model import build_model, adapt_position_encoding
from src.transforms import clip_transform
from sacred import Experiment
ex = Experiment("CIFAR")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

class CIFARDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self._config = _config
        self.transforms = clip_transform(_config['image_size'])

    def prepare_data(self):
        data_root = self._config['data_root']
        if self._config["group_name"] == 'cifar10':
            CIFAR10(root=f'{data_root}/cifar10',train=True,download=True, transform=self.transforms)
            CIFAR10(root=f'{data_root}/cifar10',train=False,download=True, transform=self.transforms)
        elif self._config["group_name"] == 'cifar100':
            CIFAR100(root=f'{data_root}/cifar100',train=True,download=True, transform=self.transforms)
            CIFAR100(root=f'{data_root}/cifar100',train=False,download=True, transform=self.transforms)

    def setup(self, stage):
        data_root = self._config['data_root']
        if self._config["group_name"] == 'cifar10':
            self.cifar_train = CIFAR10(root=f'{data_root}/cifar10',train=True,download=True, transform=self.transforms)
            self.cifar_test = CIFAR10(root=f'{data_root}/cifar10',train=False,download=True, transform=self.transforms)
            self.num_labels = 10
        elif self._config["group_name"] == 'cifar100':
            self.cifar_train = CIFAR100(root=f'{data_root}/cifar100',train=True,download=True, transform=self.transforms)
            self.cifar_test = CIFAR100(root=f'{data_root}/cifar100',train=False,download=True, transform=self.transforms)
            self.num_labels = 100

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self._config["per_gpu_batchsize"], shuffle=True, num_workers=self._config["num_workers"])
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self._config["per_gpu_eval_batchsize"], shuffle=False, num_workers=self._config["num_workers"])
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self._config["per_gpu_eval_batchsize"], shuffle=False, num_workers=self._config["num_workers"])


class CLIPViTModule(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        load_path: str = None,
        image_size: int = 224,
        hidden_size: int = 768,
        patch_size: int = 16,
        resolution_before: int = 224,
        vit_remove_last: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_labels = num_labels
        self.model = build_model(model_name_or_path, resolution_after=image_size, model_type="ViT", vit_remove_last=vit_remove_last)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        self.the_metric = -1

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = {k.replace('vit_model.', ''): v for k, v in state_dict.items() if k.startswith("vit_model")}
            if resolution_before != image_size:
                state_dict = adapt_position_encoding(state_dict, after=image_size, patch_size=patch_size)
            self.model.load_state_dict(state_dict, strict=False)
        
        self.metric = datasets.load_metric('accuracy', experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    def infer(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.classifier(self.model(inputs)[:, 0, :])
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        predictions = logits.argmax(-1)

        return loss, predictions

    def training_step(self, batch, batch_idx):
        loss, _ = self.infer(batch, batch_idx)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_loss", torch.stack([x["loss"] for x in outs]).mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, predictions = self.infer(batch, batch_idx)

        return {"loss": loss, "preds": predictions, "labels": batch[1]}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics_results = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics_results, prog_bar=True)
        self.the_metric = max(self.the_metric, metrics_results['accuracy'])
        self.log("the_metric", self.the_metric)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon, betas=(0.9, 0.98))

        self.total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs // self.trainer.accumulate_grad_batches // max(1, self.trainer.gpus)
        
        print(self.total_steps)
        print(self.hparams.warmup_steps if type(self.hparams.warmup_steps) is int else self.hparams.warmup_steps * self.total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps if type(self.hparams.warmup_steps) is int else self.hparams.warmup_steps * self.total_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


@ex.config
def config():
    root_dir = "."
    data_root = f"{root_dir}/dataset/cifar"
    log_dir = f"{root_dir}/logs"
    output_dir = f"{root_dir}/checkpoints"
    load_path = f""
    load_flag = False # load from load_path or clip-vit
    num_gpus = 8
    num_nodes = 1
    num_workers = 8
    precision = 32
    per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#
    per_gpu_eval_batchsize = 256

    # Wandb Logger Setting
    exp_name = "Uni-Modal"
    group_name = "cifar10"
    run_name = "finetune"
    
    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    log_every_n_steps = 50

    # Experiment Setting
    seed = 0
    batch_size = 512  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    vit = 'CLIP-ViT-B/16'
    image_size = 224 # 32?
    patch_size = 16
    resolution_before = 224
    input_image_embed_size = 768
    vit_remove_last = False


    # Optimizer Setting
    learning_rate = 2e-5 # 0.03 for ViT-B/16
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_epoch = 10
    max_steps = -1 # 10000 for ViT-B/16
    warmup_steps = 0.06 # 0.05 for ViT-B/16
    patience = 3

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    # pl.seed_everything(_config["seed"])

    dm = CIFARDataModule(_config)
    dm.setup("fit")
    
    model = CLIPViTModule(
        model_name_or_path=_config["vit"], 
        load_path=_config["load_path"] if _config["load_flag"] else None,
        num_labels=dm.num_labels,
        learning_rate=_config["learning_rate"],
        warmup_steps=_config["warmup_steps"],
        weight_decay=_config["weight_decay"],
        adam_epsilon=_config["adam_epsilon"],
        train_batch_size=_config["per_gpu_batchsize"],
        eval_batch_size=_config["per_gpu_eval_batchsize"],
        image_size=_config["image_size"],
        hidden_size=_config["input_image_embed_size"],
        patch_size=_config["patch_size"],
        resolution_before=_config["resolution_before"],
        vit_remove_last=_config["vit_remove_last"],
    )

    exp_name = _config["exp_name"]
    group_name = _config["group_name"]
    run_name = _config["run_name"]
    output_dir = f'{_config["output_dir"]}/{exp_name}_{group_name}_{run_name}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = WandbLogger(save_dir=_config["log_dir"], project=exp_name, name=f'{exp_name}_{group_name}_{run_name}', group=group_name)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # early_stop_callback = pl.callbacks.EarlyStopping(
    #     monitor='the_metric',
    #     patience=_config["patience"],
    #     strict=True,
    #     verbose=True,
    #     mode='max'
    # )
    # callbacks = [lr_callback, early_stop_callback]
    callbacks = [lr_callback]
    logger.log_hyperparams(_config)
    
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if _config["max_steps"] == -1 else 1000,
        max_steps=_config["max_steps"],
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=_config["log_every_n_steps"],
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        callbacks=callbacks,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
