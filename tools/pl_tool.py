import torch
import lightning.pytorch as pl


torch.set_float32_matmul_precision("high")


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.data.dtype)

    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate  # 学习率
        self.len_trainloader = len_trainloader  # 训练数据加载器长度
        self.opt = opt  # 配置参数
        self.model = model  # 模型
        self.loss1 = torch.nn.CrossEntropyLoss()

        self.use_ema = getattr(opt, "use_ema", True)
        self.ema_decay = getattr(opt, "ema_decay", 0.999)
        if self.use_ema:
            self.ema = None  # 延迟初始化EMA，等到模型参数部署到适当设备后
            self.ema_initialized = False

    def _init_ema(self):
        """初始化EMA"""
        if self.use_ema and not self.ema_initialized:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema_initialized = True

    def forward(self, x):
        """前向传播"""
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """配置优化器和学习率 Scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.len_trainloader
            * self.opt.epochs
            // len(self.opt.devices),  # 多卡训练时 steps 会混乱
            pct_start=self.opt.pct_start,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        if self.use_ema and not self.ema_initialized:
            self._init_ema()
        image, label = (batch["image"], batch["label"])
        prediction = self(image)  # 前向传播
        ce_loss = self.loss1(prediction, label)  # 计算交叉熵损失
        loss = ce_loss
        self.log("loss/train_ce_loss", ce_loss)  # 记录训练交叉熵损失
        self.log("loss/train_loss", loss)  # 记录训练损失
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])

        # 更新EMA参数
        if self.use_ema:
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        image, label = (batch["image"], batch["label"])
        prediction = self(image)  # 前向传播
        ce_loss = self.loss1(prediction, label)  # 计算交叉熵损失
        loss = ce_loss
        self.log("loss/valid_ce_loss", ce_loss)  # 记录验证交叉熵损失
        self.log("loss/valid_loss", loss)  # 记录验证损失

    def on_validation_start(self):
        """验证开始时应用EMA参数"""
        if self.use_ema and self.ema_initialized:
            self.ema.apply_shadow()

    def on_validation_end(self):
        """验证结束时恢复原始参数"""
        if self.use_ema and self.ema_initialized:
            self.ema.restore()

    def on_train_epoch_end(self):
        """训练周期结束时执行"""
        pass

    def on_validation_epoch_end(self):
        """验证周期结束时执行"""
        pass

    def on_save_checkpoint(self, checkpoint):
        """保存checkpoint时保存EMA参数"""
        if self.use_ema and self.ema_initialized:
            # 统一使用float32存储EMA状态
            checkpoint["ema_state_dict"] = {
                k: v.clone().float().cpu() for k, v in self.ema.shadow.items()
            }

    def on_load_checkpoint(self, checkpoint):
        """加载checkpoint时加载EMA参数"""
        if self.use_ema and "ema_state_dict" in checkpoint:
            self._init_ema()
            # 加载已保存的EMA状态
            for k, v in checkpoint["ema_state_dict"].items():
                if k in self.ema.shadow:
                    self.ema.shadow[k] = v.to(self.device)
