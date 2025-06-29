"""
PyTorch Lightning训练工具模块

这个文件包含了深度学习训练的核心组件：
1. EMA (Exponential Moving Average) 类：实现指数移动平均，提升模型性能和稳定性
2. LightningModule 类：基于PyTorch Lightning的训练模块，封装了完整的训练流程

主要功能：
- 自动化训练循环（训练、验证、测试）
- 指数移动平均参数更新
- 学习率调度和优化器配置
- 检查点保存和加载
- 训练指标记录和监控


=== 如何自定义和修改 ===

1. 修改损失函数：
   - 在LightningModule.__init__中修改self.loss1
   - 添加新的损失函数（如Focal Loss, Label Smoothing等）
   - 在training_step和validation_step中计算新损失

2. 更换优化器：
   - 在configure_optimizers中修改优化器类型
   - 支持SGD, Adam, AdamW, RMSprop等
   - 调整学习率调度策略

3. 添加评估指标：
   - 在validation_step中计算accuracy, F1-score等
   - 使用torchmetrics库的现成指标
   - 自定义评估指标

4. 自定义数据预处理：
   - 修改training_step中的数据解析部分
   - 支持多输入、多输出的模型
   - 处理不同的数据格式

5. 扩展回调功能：
   - 在on_train_epoch_end中添加学习率记录
   - 在on_validation_epoch_end中添加早停逻辑
   - 添加自定义的训练监控

6. EMA参数调整：
   - 修改ema_decay值（0.99-0.9999）
   - 关闭EMA（设置use_ema=False）
   - 自定义EMA更新策略
"""

import torch
import lightning.pytorch as pl
from .model_registry import get_model

# 设置PyTorch矩阵乘法精度为高精度模式，提升性能
torch.set_float32_matmul_precision("high")


class EMA:
    """
    指数移动平均 (Exponential Moving Average) 类

    EMA是一种模型参数平滑技术，通过维护参数的移动平均来提升模型性能：
    - 减少训练过程中的参数震荡
    - 提升模型的泛化能力
    - 在验证时使用平滑后的参数，训练时使用原始参数

    工作原理：
    shadow_param = decay * shadow_param + (1 - decay) * current_param

    Args:
        model: 需要应用EMA的PyTorch模型
        decay: 衰减系数，越接近1越平滑，通常设置为0.999
    """

    def __init__(self, model, decay=0.999):
        self.model = model  # 原始模型
        self.decay = decay  # EMA衰减系数
        self.shadow = {}  # 存储EMA参数的字典
        self.backup = {}  # 临时备份原始参数
        self.register()  # 初始化EMA参数

    def register(self):
        """
        注册模型参数到EMA系统
        将所有需要梯度的参数复制到shadow字典中作为初始EMA参数
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        更新EMA参数
        在每个训练步骤后调用，使用当前参数更新EMA参数
        公式: shadow = decay * shadow + (1 - decay) * current
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        将EMA参数应用到模型
        通常在验证开始时调用，使用平滑后的参数进行推理
        同时备份原始参数以便后续恢复
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()  # 备份原始参数
                param.data = self.shadow[name].to(param.data.dtype)  # 应用EMA参数

    def restore(self):
        """
        恢复原始模型参数
        通常在验证结束时调用，恢复训练用的原始参数
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}  # 清空备份


class LightningModule(pl.LightningModule):
    """
    基于PyTorch Lightning的训练模块

    这个类封装了深度学习模型的完整训练流程，包括：
    - 模型前向传播和损失计算
    - 优化器和学习率调度器配置
    - 训练和验证步骤的定义
    - EMA参数管理
    - 检查点保存和加载
    - 训练指标的记录和监控

    主要特性：
    - 支持动态模型加载或使用预定义模型
    - 集成EMA优化技术
    - 使用OneCycleLR学习率调度
    - 自动化的训练和验证循环

    Args:
        opt: 配置对象，包含所有训练参数
        model: 预定义的模型，如果为None则动态加载
        len_trainloader: 训练数据加载器的长度，用于学习率调度
    """

    def __init__(self, opt, model, len_trainloader):
        super().__init__()

        # ==================== 基础配置 ====================
        self.learning_rate = opt.learning_rate  # 学习率
        self.len_trainloader = len_trainloader  # 训练数据加载器长度
        self.opt = opt  # 配置参数对象

        # ==================== 模型初始化 ====================
        # 支持两种模型加载方式：动态加载或使用传入的模型
        if model is None:
            # 动态模型加载：根据配置文件创建模型
            model_kwargs = {
                "input_channels": getattr(
                    opt, "in_chans", 3
                ),  # 输入通道数，默认3（RGB）
                "num_classes": getattr(opt, "num_classes", 10),  # 分类数量，默认10
            }
            # 过滤掉 None 值的参数，避免传递无效参数
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

            self.model = get_model(opt.model_name, **model_kwargs)
            print(f"动态加载模型: {opt.model_name}")
        else:
            # 使用传入的预定义模型
            self.model = model

        # ==================== 损失函数定义 ====================
        # 交叉熵损失，适用于多分类任务
        #
        # 如何自定义损失函数：
        # 1. 回归任务：self.loss1 = torch.nn.MSELoss() 或 torch.nn.L1Loss()
        # 2. 二分类任务：self.loss1 = torch.nn.BCEWithLogitsLoss()
        # 3. 多标签分类：self.loss1 = torch.nn.MultiLabelSoftMarginLoss()
        # 4. 自定义损失：from tools.losses import FocalLoss; self.loss1 = FocalLoss()
        # 5. 组合损失：可以定义多个损失函数并在训练中组合使用
        #
        # 示例：添加辅助损失
        # self.loss1 = torch.nn.CrossEntropyLoss()  # 主要损失
        # self.loss2 = torch.nn.MSELoss()           # 辅助损失（如特征重建）
        self.loss1 = torch.nn.CrossEntropyLoss()

        # ==================== EMA配置 ====================
        # 指数移动平均配置，用于提升模型性能
        #
        # EMA参数调整指南：
        # - use_ema: 是否启用EMA，建议在稳定训练后启用
        # - ema_decay: 衰减系数，影响平滑程度
        #   * 0.99: 较快更新，适合小模型或快速实验
        #   * 0.999: 标准设置，适合大多数情况
        #   * 0.9999: 极慢更新，适合大模型或长期训练
        #
        # 何时使用EMA：
        # - 模型参数震荡较大时
        # - 需要更稳定的验证性能时
        # - 生产环境部署时（使用EMA权重）
        #
        # 禁用EMA：在配置文件中设置 use_ema: false
        self.use_ema = getattr(opt, "use_ema", True)  # 是否使用EMA
        self.ema_decay = getattr(opt, "ema_decay", 0.999)  # EMA衰减系数
        if self.use_ema:
            self.ema = None  # 延迟初始化EMA
            self.ema_initialized = False  # EMA初始化标志

    def _init_ema(self):
        """
        延迟初始化EMA
        在模型参数部署到适当设备后才初始化EMA，确保设备一致性
        """
        if self.use_ema and not self.ema_initialized:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema_initialized = True

    def forward(self, x):
        """
        模型前向传播

        Args:
            x: 输入张量，通常是图像数据

        Returns:
            pred: 模型预测输出，通常是分类logits
        """
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        这个方法定义了训练过程中使用的优化算法和学习率变化策略：
        - 优化器：AdamW，具有权重衰减的Adam优化器
        - 调度器：OneCycleLR，实现学习率的单周期变化

        Returns:
            dict: 包含优化器和调度器配置的字典

        === 自定义优化器示例 ===

        1. 使用SGD优化器：
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.opt.weight_decay
        )

        2. 使用Adam优化器：
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.opt.weight_decay
        )

        3. 分层学习率（不同层使用不同学习率）：
        backbone_params = [p for n, p in self.model.named_parameters() if 'backbone' in n]
        head_params = [p for n, p in self.model.named_parameters() if 'head' in n]
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},
            {'params': head_params, 'lr': self.learning_rate}
        ])

        === 自定义学习率调度器示例 ===

        1. 指数衰减：
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.95
        )

        2. 余弦退火：
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.opt.epochs
        )

        3. 阶梯式衰减：
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

        4. ReduceLROnPlateau（基于验证指标）：
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        # 注意：需要在返回字典中设置 "monitor": "loss/valid_loss"
        """
        # AdamW优化器：结合了Adam的自适应学习率和L2权重衰减
        self.optimizer = torch.optim.AdamW(
            self.parameters(),  # 模型参数
            weight_decay=self.opt.weight_decay,  # L2正则化系数
            lr=self.learning_rate,  # 初始学习率
        )

        # OneCycleLR调度器：实现学习率的单周期变化
        # 先上升到最大值，然后下降，有助于快速收敛和跳出局部最优
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,  # 最大学习率
            total_steps=self.len_trainloader * self.opt.epochs,  # 总训练步数
            pct_start=self.opt.pct_start,  # 上升阶段占比
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",  # 每步更新学习率（也可以是"epoch"）
                # "monitor": "loss/valid_loss",  # 用于ReduceLROnPlateau等调度器
                # "frequency": 1,               # 更新频率
            },
        }

    def training_step(self, batch, batch_idx):
        """
        单个训练步骤

        这个方法定义了每个训练批次的处理流程：
        1. 延迟初始化EMA（确保设备一致性）
        2. 解析输入数据
        3. 前向传播计算预测
        4. 计算损失函数
        5. 记录训练指标
        6. 更新EMA参数

        Args:
            batch: 一个批次的数据，包含image和label
            batch_idx: 当前批次的索引

        Returns:
            loss: 训练损失，用于反向传播
        """
        # 延迟初始化EMA，确保模型已经部署到正确设备
        if self.use_ema and not self.ema_initialized:
            self._init_ema()

        # 解析批次数据
        image, label = (batch["image"], batch["label"])

        # 前向传播：获取模型预测
        prediction = self(image)

        # 计算交叉熵损失
        ce_loss = self.loss1(prediction, label)
        loss = ce_loss

        # 记录训练指标到wandb/tensorboard
        self.log("loss/train_ce_loss", ce_loss)  # 训练交叉熵损失
        self.log("loss/train_loss", loss)  # 总训练损失
        self.log(
            "trainer/learning_rate", self.optimizer.param_groups[0]["lr"]
        )  # 当前学习率

        # 更新EMA参数（在每个训练步骤后）
        if self.use_ema:
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        """
        单个验证步骤

        这个方法定义了每个验证批次的处理流程：
        1. 解析输入数据
        2. 前向传播（注意：此时使用的是EMA参数）
        3. 计算损失函数
        4. 记录验证指标

        注意：验证时模型使用EMA参数，这通常能获得更好的性能

        Args:
            batch: 一个批次的验证数据
            batch_idx: 当前批次的索引
        """
        # 解析批次数据
        image, label = (batch["image"], batch["label"])

        # 前向传播：使用EMA参数进行推理
        prediction = self(image)

        # 计算交叉熵损失
        ce_loss = self.loss1(prediction, label)
        loss = ce_loss

        # 记录验证指标
        self.log("loss/valid_ce_loss", ce_loss)  # 验证交叉熵损失
        self.log("loss/valid_loss", loss)  # 总验证损失

    def on_validation_start(self):
        """
        验证开始时的钩子函数

        在每次验证开始前自动调用，将模型参数切换为EMA参数
        这样验证时使用的是平滑后的参数，通常能获得更好的性能
        """
        if self.use_ema and self.ema_initialized:
            self.ema.apply_shadow()

    def on_validation_end(self):
        """
        验证结束时的钩子函数

        在每次验证结束后自动调用，恢复原始的训练参数
        确保训练过程使用的是正常的参数，而不是EMA参数
        """
        if self.use_ema and self.ema_initialized:
            self.ema.restore()

    def on_train_epoch_end(self):
        """
        训练周期结束时的钩子函数

        在每个训练epoch结束时调用，可以在这里添加：
        - 模型评估代码
        - 额外的日志记录
        - 学习率调整
        - 其他周期性操作

        当前为空实现，可根据需要扩展
        """
        pass

    def on_validation_epoch_end(self):
        """
        验证周期结束时的钩子函数

        在每个验证epoch结束时调用，可以在这里添加：
        - 验证结果汇总
        - 早停判断
        - 模型选择逻辑
        - 其他验证后处理

        当前为空实现，可根据需要扩展
        """
        pass

    def on_save_checkpoint(self, checkpoint):
        """
        保存检查点时的钩子函数

        在保存模型检查点时自动调用，除了保存模型参数外，还保存EMA状态
        这样在恢复训练时可以完整恢复EMA的状态

        Args:
            checkpoint: 检查点字典，包含模型状态、优化器状态等
        """
        if self.use_ema and self.ema_initialized:
            # 将EMA参数转换为float32并移到CPU，统一存储格式
            # 这样可以避免不同精度和设备之间的兼容性问题
            checkpoint["ema_state_dict"] = {
                k: v.clone().float().cpu() for k, v in self.ema.shadow.items()
            }

    def on_load_checkpoint(self, checkpoint):
        """
        加载检查点时的钩子函数

        在加载模型检查点时自动调用，恢复EMA状态以继续训练
        确保从检查点恢复后EMA能够正常工作

        Args:
            checkpoint: 包含模型状态和EMA状态的检查点字典
        """
        if self.use_ema and "ema_state_dict" in checkpoint:
            # 初始化EMA（如果还未初始化）
            self._init_ema()
            # 恢复保存的EMA状态，并确保设备一致性
            for k, v in checkpoint["ema_state_dict"].items():
                if k in self.ema.shadow:
                    self.ema.shadow[k] = v.to(self.device)
