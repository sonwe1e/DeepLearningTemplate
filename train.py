import torch
from configs.option import get_option, set_default_config_path
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from tools.pl_tool import LightningModule
import wandb


torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # 设置配置文件路径 - 这样所有后续的 get_option 调用都会使用这个路径
    set_default_config_path("/media/hdd/sonwe1e/Template/configs/config.yaml")

    # 现在可以直接调用 get_option()，它会使用上面设置的路径
    opt, checkpoint_path = get_option()

    # 导入数据集相关模块 - 现在它们会使用正确的配置路径
    from tools.datasets.datasets import *
    from tools.model_registry import list_available_models, get_model

    # 设置随机种子
    pl.seed_everything(opt.seed)

    # 打印可用模型
    print("可用模型列表:")
    for model_name in list_available_models():
        print(f"  - {model_name}")

    # 获取数据加载器
    """定义网络"""
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
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

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=opt.resume,
    )
    wandb.finish()
