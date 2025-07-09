"""深度学习模型训练主程序"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import os

from configs.option import get_option, set_default_config_path
from tools.model_registry import get_model, list_available_models
from tools.pl_tool import LightningModule
from tools.datasets.datasets import get_dataloader

torch.set_float32_matmul_precision("high")


def main():
    set_default_config_path("/media/hdd/sonwe1e/Template/configs/config.yaml")
    opt, checkpoint_path = get_option()

    pl.seed_everything(opt.seed)

    print("可用模型列表:")
    for model_name in list_available_models():
        print(f"  - {model_name}")

    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    train_dataloader, valid_dataloader = get_dataloader(opt)

    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.devices,
        strategy="auto",
        max_epochs=opt.epochs,
        precision=opt.precision,
        default_root_dir="./",
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=opt.log_step,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        gradient_clip_val=opt.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(checkpoint_path, "./checkpoints"),
                monitor=f"loss/{opt.save_metric}",
                mode="min",
                save_top_k=opt.save_checkpoint_num,
                save_last=False,
                filename="epoch_{epoch}-loss_{loss/valid_loss:.3f}",
                auto_insert_metric_name=False,
            )
        ],
    )

    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=opt.resume,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
