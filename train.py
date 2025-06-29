"""
深度学习模型训练主程序

这个文件是整个项目的训练入口点，主要功能包括：
1. 配置管理：加载训练配置和超参数
2. 模型初始化：根据配置创建指定的深度学习模型
3. 数据准备：设置训练和验证数据加载器
4. 训练流程：使用PyTorch Lightning进行模型训练
5. 实验跟踪：集成Wandb进行训练过程监控和日志记录
6. 模型保存：自动保存最佳检查点

使用方法：
    python train.py

注意：
- 确保配置文件 configs/config.yaml 存在且配置正确
- 训练数据路径在配置文件中正确设置
- 根据硬件情况调整批次大小和设备配置

=== 如何自定义训练 ===

1. 修改配置文件 (configs/config.yaml):
   - 调整学习率、批次大小、训练轮数等超参数
   - 设置数据路径和模型保存路径
   - 配置模型类型和参数

2. 添加新模型:
   - 在 tools/models/ 目录下创建新的模型文件
   - 在 tools/model_registry.py 中注册新模型
   - 在配置文件中指定新模型名称

3. 自定义数据集:
   - 修改 tools/datasets/datasets.py 中的Dataset类
   - 实现真实数据的加载逻辑
   - 调整数据增强策略

4. 添加新的损失函数:
   - 在 tools/pl_tool.py 的LightningModule中修改loss计算
   - 或在 tools/losses.py 中定义新的损失函数

5. 自定义训练回调:
   - 添加学习率调度、早停等回调函数
   - 在trainer的callbacks列表中添加新的回调

6. 调试和监控:
   - 检查wandb日志了解训练进度
   - 调整验证频率和日志记录频率
   - 使用检查点恢复中断的训练
"""

import torch
from configs.option import get_option, set_default_config_path
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from tools.pl_tool import LightningModule
import wandb

# 设置PyTorch矩阵乘法精度为高精度模式，可以提升在某些硬件上的性能
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # ==================== 配置初始化 ====================
    # 设置默认配置文件路径，后续所有 get_option() 调用都会使用这个路径
    # 这样可以避免每次调用时都需要传递配置文件路径
    #
    # 自定义配置文件：
    # - 复制 configs/config.yaml 创建新的配置文件
    # - 修改路径指向新的配置文件
    # - 例如：set_default_config_path("./my_custom_config.yaml")
    set_default_config_path("/media/hdd/sonwe1e/Template/configs/config.yaml")

    # 加载训练配置和选项，同时获取检查点保存路径
    # opt: 包含所有训练参数的配置对象
    # checkpoint_path: 模型检查点的保存路径
    #
    # 配置对象包含的主要参数：
    # - opt.learning_rate: 学习率
    # - opt.epochs: 训练轮数
    # - opt.train_batch_size / opt.valid_batch_size: 批次大小
    # - opt.model: 模型配置字典
    # - opt.data_path: 数据路径
    # - opt.devices: 使用的GPU数量
    opt, checkpoint_path = get_option()

    # ==================== 模块导入 ====================
    # 导入数据集相关模块 - 现在它们会使用上面设置的配置路径
    # 注意：这里使用通配符导入是为了动态加载数据集类
    #
    # 如何添加自定义数据集：
    # 1. 在 tools/datasets/ 目录下创建新的数据集文件
    # 2. 实现自定义的Dataset类，继承torch.utils.data.Dataset
    # 3. 在 datasets.py 中导入并注册新的数据集类
    # 4. 修改 get_dataloader 函数来使用新的数据集
    from tools.datasets.datasets import *
    from tools.model_registry import list_available_models, get_model

    # ==================== 随机种子设置 ====================
    # 设置全局随机种子，确保实验结果的可重现性
    # 这会影响PyTorch、NumPy、Python等的随机数生成

    pl.seed_everything(opt.seed)

    # ==================== 模型选择与初始化 ====================
    # 打印所有可用的模型列表，方便用户了解可选择的模型
    #
    # 如何添加新模型：
    # 1. 在 tools/models/ 目录下创建新的模型文件（如 my_model.py）
    # 2. 在配置文件中设置 model.model_name 为新模型名称
    # 3. 配置 model.model_kwargs 中的模型参数
    print("可用模型列表:")
    for model_name in list_available_models():
        print(f"  - {model_name}")

    # 根据配置文件中的设置创建指定的深度学习模型
    # model_name: 模型的名称
    # model_kwargs: 模型的具体参数
    #
    # 配置示例（在config.yaml中）：
    # model:
    #   model_name: "ResNet"
    #   model_kwargs:
    #     num_classes: 10
    #     layers: [2, 2, 2, 2]
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])

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
