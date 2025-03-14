import torch
from configs.option import get_option
from tools.datasets.datasets import *
from tools.pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import os

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option()
    """定义网络"""
    import timm

    model = timm.create_model(
        opt.model_name,
        num_classes=opt.num_classes,
        in_chans=opt.in_chans,
        pretrained=opt.pretrained,
    )
    """模型编译"""
    # model = torch.compile(model)
    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt)

    """Lightning 模块定义"""
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
                dirpath=os.path.join("./checkpoints", opt.exp_name),
                monitor=f"loss/{opt.save_metric}",
                mode="min",
                save_top_k=opt.save_checkpoint_num,
                save_last=False,
                filename="epoch_{epoch}-loss_{loss/valid_loss:.3f}",
                auto_insert_metric_name=False,
            )
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=opt.resume,
    )
    wandb.finish()
