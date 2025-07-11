import torch
from configs.option import get_option, set_default_config_path
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from tools.pl_tool import LightningModule
import wandb

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    set_default_config_path("./configs/config.yaml")

    opt, checkpoint_path = get_option()

    from tools.datasets.datasets import *
    from tools.model_registry import list_available_models, get_model

    pl.seed_everything(opt.seed)

    print("可用模型列表:")
    for model_name in list_available_models():
        print(f"  - {model_name}")

    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    #  model.load_from_checkpoint(
    #      "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/baselinev1_2025-07-09_20-17-28/checkpoints/epoch_49-loss_0.386.ckpt"
    #  )

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
                monitor=f"metrics/{opt.save_metric}",
                mode="max",
                save_top_k=opt.save_checkpoint_num,
                save_last=False,
                filename="epoch_{epoch}-mIoU_{metrics/val_miou:.3f}",
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
    #   model_kwargs:
    #     num_classes: 10
    #     layers: [2, 2, 2, 2]
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    #  model.load_from_checkpoint(
    #      "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/large_pre.ckpt"
    #  )  # 加载预训练模型配置

    # 可选：模型编译优化（当前被注释掉）
    # 在某些情况下可以提升训练速度，但可能增加内存使用
    # PyTorch 2.0+ 支持，可以在较新的硬件上获得性能提升
    # 启用方法：取消下一行的注释
    # model = torch.compile(model)

    # ==================== 数据加载器准备 ====================
    # 根据配置创建训练和验证数据加载器
    # 这些加载器会自动处理数据的批处理、洗牌、数据增强等
    #
    # 数据加载器配置要点：
    # - train_batch_size / valid_batch_size: 根据GPU显存调整
    # - num_workers: 数据加载的并行进程数，通常设为CPU核心数的1-2倍
    # - 数据增强在 tools/datasets/augments.py 中配置
    #
    # 如何使用自定义数据：
    # 1. 修改配置文件中的 data_path 指向你的数据目录
    # 2. 在 tools/datasets/datasets.py 中实现数据加载逻辑
    # 3. 确保数据目录结构符合Dataset类的要求
    train_dataloader, valid_dataloader = get_dataloader(opt)

    # ==================== 实验跟踪设置 ====================
    # 配置Weights & Biases (wandb) 日志记录器
    # 用于实时监控训练过程、记录指标、保存模型等
    #
    # Wandb配置说明：
    # - project: 项目名称，可以在配置文件中修改
    # - name: 实验名称，建议包含模型名称和关键参数
    # - offline: 设为True可离线运行，日志保存在本地
    # - 首次使用需要注册wandb账号并登录：wandb login
    #
    # 如果不想使用wandb：
    # - 在配置文件中设置 save_wandb: false
    # - 或者将logger参数设为None禁用日志记录
    wandb_logger = WandbLogger(
        project=opt.project,  # 项目名称，用于在wandb中组织实验
        name=opt.exp_name,  # 实验名称，每次运行的唯一标识
        offline=not opt.save_wandb,  # 是否离线模式（当网络不可用时）
        config=opt,  # 将所有配置参数上传到wandb
    )

    # ==================== 训练器配置 ====================
    # 创建PyTorch Lightning训练器，集成了训练循环的所有功能
    trainer = pl.Trainer(
        # 硬件配置
        accelerator="auto",  # 自动检测可用的加速器
        devices=opt.devices,  # 使用的设备数量
        strategy="auto",  # 分布式训练策略
        # 训练配置
        max_epochs=opt.epochs,  # 最大训练轮数
        precision=opt.precision,  # 数值精度(如混合精度训练)
        # 路径和日志配置
        default_root_dir="./",  # 默认保存路径
        logger=wandb_logger,  # 使用wandb记录器
        # 验证和日志频率
        val_check_interval=opt.val_check,  # 验证检查间隔
        log_every_n_steps=opt.log_step,  # 每N步记录一次日志
        # 训练优化配置
        accumulate_grad_batches=opt.accumulate_grad_batches,  # 梯度累积步数
        gradient_clip_val=opt.gradient_clip_val,  # 梯度裁剪阈值
        # 回调函数配置
        callbacks=[
            # 模型检查点回调：自动保存最佳模型
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(checkpoint_path, "./checkpoints"),  # 保存路径
                monitor=f"loss/{opt.save_metric}",  # 监控的指标
                mode="min",  # 指标越小越好
                save_top_k=opt.save_checkpoint_num,  # 保存最好的K个检查点
                save_last=False,  # 不保存最后一个检查点
                filename="epoch_{epoch}-loss_{loss/valid_loss:.3f}",  # 文件命名格式
                auto_insert_metric_name=False,  # 不自动插入指标名称
            )
        ],
    )

    # ==================== 开始训练 ====================
    # 启动训练流程
    # LightningModule: 包装了模型、损失函数、优化器等的Lightning模块
    # train_dataloaders: 训练数据加载器
    # val_dataloaders: 验证数据加载器
    # ckpt_path: 如果指定，则从该检查点恢复训练
    #
    # 训练恢复：
    # - 在配置文件中设置 resume 参数为检查点路径
    # - 例如：resume: "./experiments/my_exp/checkpoints/epoch_10-loss_0.123.ckpt"
    # - 恢复训练会保持原有的优化器状态和学习率调度
    #
    # 常见问题：
    # - 如果训练中断，会自动从最新检查点恢复
    # - 如果显存溢出，减小batch_size或启用梯度累积
    # - 如果收敛慢，检查学习率设置和数据质量
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),  # Lightning模块实例
        train_dataloaders=train_dataloader,  # 训练数据
        val_dataloaders=valid_dataloader,  # 验证数据
        ckpt_path=opt.resume,  # 恢复训练的检查点路径
    )

    # ==================== 训练结束清理 ====================
    # 结束wandb会话，确保所有日志都被正确上传和保存
    #
    # 训练完成后的工作：
    # 1. 查看wandb dashboard中的训练结果
    # 2. 检查保存的最佳模型检查点
    # 3. 进行模型测试和评估
    # 4. 分析训练日志，为下次实验做准备
    #
    # 下一步操作：
    # - 使用保存的检查点进行推理或部署
    # - 在test.ipynb中测试模型性能
    # - 导出模型为ONNX格式（参考example_export.ipynb）
    # - 根据结果调整超参数进行新的实验
    wandb.finish()

"""
=== 使用指导和故障排除 ===

1. 首次运行准备：
   - 安装依赖：pip install -r requirements.txt
   - 配置wandb：wandb login（可选）
   - 检查GPU：nvidia-smi 确认GPU可用
   - 准备数据：确保数据路径在config.yaml中正确设置

2. 常见错误和解决方案：
   
   错误：CUDA out of memory
   解决：减小batch_size，启用混合精度(precision: 16)，增加梯度累积
   
   错误：No module named 'xxx'
   解决：pip install xxx 或检查虚拟环境激活
   
   错误：数据加载错误
   解决：检查data_path设置，确认数据文件存在
   
   错误：wandb登录失败
   解决：设置save_wandb: false 或使用wandb login登录

3. 实验管理：
   - 为每个实验设置描述性的exp_name
   - 使用git管理代码版本
   - 记录重要的实验结果和配置
   - 定期清理不需要的检查点文件

4. 调试技巧：
   - 先用小数据集验证代码正确性
   - 检查训练/验证损失趋势判断过拟合
   - 使用Learning Rate Finder寻找最佳学习率
   - 可视化数据增强结果确认数据处理正确

"""
