import os
import copy
import pytorch_lightning as pl
import wandb
import torch
import time
from pytorch_lightning.loggers import WandbLogger
import os
os.environ["NCCL_DEBUG"] = "INFO"

from src.config import ex
from src.modules import METERTransformerSS
from src.modules import BTTransformer
from src.datamodules.multitask_datamodule import MTDataModule

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config)
    if _config["model_type"] == "METER":
        model = METERTransformerSS(_config)
    elif _config["model_type"] == "BT":
        model = BTTransformer(_config)
    else:
        raise NotImplementedError("model_type {} not implemented".format(_config["model_type"]))
    
    exp_name = _config["exp_name"]
    group_name = _config["group_name"]
    run_name = _config["run_name"]
    output_dir = f'{_config["output_dir"]}/{exp_name}_{group_name}_{run_name}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True if 'irtr' in group_name else False,
        filename=f'{exp_name}' + '_{epoch:02d}_{val/the_metric:.4f}',
        auto_insert_metric_name=False,
        dirpath=output_dir,
    )

    logger = WandbLogger(save_dir=_config["log_dir"], project=exp_name, name=f'{exp_name}_{group_name}_{run_name}', group=group_name)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"]

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps == -1 else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=_config["log_every_n_steps"],
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        best_metric_log = model.best_metric_log
        best_model_path = checkpoint_callback.best_model_path
        print(f'best_model_path: {best_model_path}')
        if _config["group_name"] in ["irtr_coco", "irtr_f30k"]: # choose the last checkpoint for test evaluation
            best_model_path = checkpoint_callback.last_model_path
            print(f'last_model_path: {checkpoint_callback.last_model_path}')

        # Directly running test evaluation
        if _config["group_name"] not in ["mlm_itm", "nlvr2", "snli", "irtr_itm_itc_f30k", "irtr_itm_itc_coco"]: # these tasks do not need to run the test evaluation after training.
            # Remember: Here you need to transfer the best model checkpoint to each node. For example, the node-0 upload the best checkpoint and the node-1 and node-2 download the best checkpoint.

            test_config = copy.deepcopy(_config)
            test_config["load_path"] = best_model_path
            test_config["test_only"] = True
            if test_config["group_name"] in ["irtr_coco", "irtr_f30k"]:
                test_config["get_recall_metric"] = True
            test_dm = MTDataModule(test_config)
            if test_config["model_type"] == "METER":
                test_model = METERTransformerSS(test_config)
            elif test_config["model_type"] == "BT":
                test_model = BTTransformer(test_config)
            trainer.test(test_model, datamodule=test_dm)
            if _config["group_name"] not in ["vqa"]:
                best_metric_log.update(test_model.best_metric_log)
        
        logger.log_text(key="best_metrics", columns=list(best_metric_log.keys()), data=[list(best_metric_log.values())])
    else:
        trainer.test(model, datamodule=dm)
