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
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sacred import Experiment
ex = Experiment("GLUE")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

glue_the_metric = {
    'cola': 'matthews_correlation',
    'sst2': 'accuracy',
    'mrpc': 'f1',
    'qqp': 'f1',
    'stsb': 'spearmanr',
    'mnli': 'accuracy',
    'qnli': 'accuracy',
    'rte': 'accuracy',
    'wnli': 'accuracy',
}

# # get metric name ...
# import datasets
# task_text_field_map = {
#         "cola": ["sentence"],
#         "sst2": ["sentence"],
#         "mrpc": ["sentence1", "sentence2"],
#         "qqp": ["question1", "question2"],
#         "stsb": ["sentence1", "sentence2"],
#         "mnli": ["premise", "hypothesis"],
#         "qnli": ["question", "sentence"],
#         "rte": ["sentence1", "sentence2"],
#         "wnli": ["sentence1", "sentence2"],
#         "ax": ["premise", "hypothesis"],
# }
# for task_name in task_text_field_map:
#     print(task_name)
#     glue_metric = datasets.load_metric('glue', task_name)
#     references = [0, 1]
#     predictions = [0, 1]
#     results = glue_metric.compute(predictions=predictions, references=references)
#     print(results)

class GLUEDataModule(LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, shuffle=False)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, shuffle=False) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, shuffle=False)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, shuffle=False) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        load_path: str = None,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.the_metric = -1
        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = {k.replace('text_transformer.', ''): v for k, v in state_dict.items() if k.startswith("text_transformer")}
            self.model.roberta.load_state_dict(state_dict, strict=False)
        
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            accumulate_the_metric = 0
            accumulate_counts = 0
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
                accumulate_the_metric += list(split_metrics.values())[0]
                accumulate_counts += 1
            self.the_metric = max(self.the_metric, accumulate_the_metric / accumulate_counts)
            self.log("the_metric", self.the_metric)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics_results = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics_results, prog_bar=True)
        the_metric_name = glue_the_metric[self.hparams.task_name]
        self.the_metric = max(self.the_metric, metrics_results[the_metric_name])
        self.log("the_metric", self.the_metric)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        self.total_steps = len(train_loader.dataset) * self.trainer.max_epochs // self.hparams.train_batch_size // self.trainer.accumulate_grad_batches // max(1, self.trainer.gpus)

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
    data_root = f"{root_dir}/dataset/glue"
    log_dir = f"{root_dir}/logs"
    output_dir = f"{root_dir}/checkpoints"
    load_path = f""
    load_flag = False # load from load_path or roberta
    num_gpus = 1
    num_nodes = 1
    precision = 32
    per_gpu_batchsize = 32  # you should define this manually with per_gpu_batch_size=#
    per_gpu_eval_batchsize = 128

    # Wandb Logger Setting
    exp_name = "Uni-Modal"
    group_name = "cola"
    run_name = "finetune"
    
    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    log_every_n_steps = 50

    # Experiment Setting
    seed = 0
    batch_size = 32  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Text Setting
    max_seq_length = 512
    tokenizer = "roberta-base"

    # Optimizer Setting
    learning_rate = 1e-5
    weight_decay = 0.1
    adam_epsilon = 1e-6
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.06
    patience = 3

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    # pl.seed_everything(_config["seed"])

    dm = GLUEDataModule(
        model_name_or_path=_config["tokenizer"], 
        task_name=_config["group_name"],
        max_seq_length=_config["max_seq_length"],
        train_batch_size=_config["per_gpu_batchsize"],
        eval_batch_size=_config["per_gpu_eval_batchsize"],
    )
    dm.setup("fit")
    
    model = GLUETransformer(
        model_name_or_path=_config["tokenizer"], 
        load_path=_config["load_path"] if _config["load_flag"] else None,
        num_labels=dm.num_labels,
        learning_rate=_config["learning_rate"],
        warmup_steps=_config["warmup_steps"],
        weight_decay=_config["weight_decay"],
        adam_epsilon=_config["adam_epsilon"],
        train_batch_size=_config["per_gpu_batchsize"],
        eval_batch_size=_config["per_gpu_eval_batchsize"],
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
    )

    exp_name = _config["exp_name"]
    group_name = _config["group_name"]
    run_name = _config["run_name"]
    output_dir = f'{_config["output_dir"]}/{exp_name}_{group_name}_{run_name}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = WandbLogger(save_dir=_config["log_dir"], project=exp_name, name=f'{exp_name}_{group_name}_{run_name}', group=group_name)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='the_metric',
        patience=_config["patience"],
        strict=True,
        verbose=True,
        mode='max'
    )
    callbacks = [lr_callback, early_stop_callback]
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
