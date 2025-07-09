"""PyTorch Lightning训练工具模块"""

import torch
import lightning.pytorch as pl
from .model_registry import get_model

torch.set_float32_matmul_precision("high")


class EMA:
    """指数移动平均 (Exponential Moving Average) 类"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册模型参数到EMA系统"""
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
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class LightningModule(pl.LightningModule):
    """PyTorch Lightning训练模块"""

    def __init__(self, opt, model=None, dataset_len=None):
        super().__init__()
        self.save_hyperparameters(opt, ignore=["model"])

        self.opt = opt
        self.learning_rate = opt.learning_rate
        self.dataset_len = dataset_len

        if model is None:
            model_kwargs = getattr(opt, "model_kwargs", {})
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            self.model = get_model(opt.model_name, **model_kwargs)
            print(f"动态加载模型: {opt.model_name}")
        else:
            self.model = model

        self.loss1 = torch.nn.CrossEntropyLoss()

        self.use_ema = getattr(opt, "use_ema", True)
        self.ema_decay = getattr(opt, "ema_decay", 0.999)
        if self.use_ema:
            self.ema = None
            self.ema_initialized = False

    def _init_ema(self):
        """延迟初始化EMA"""
        if self.use_ema and not self.ema_initialized:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema_initialized = True

    def forward(self, x):
        """模型前向传播"""
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=getattr(self.opt, "weight_decay", 1e-4),
            )

        if not hasattr(self, "scheduler"):
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,
                div_factor=10,
                final_div_factor=100,
            )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        self._init_ema()

        image = batch["image"]
        label = batch["label"]

        pred = self.forward(image)
        loss = self.loss1(pred, label)

        if self.use_ema and self.ema_initialized:
            self.ema.update()

        self.log("loss/train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        image = batch["image"]
        label = batch["label"]

        if self.use_ema and self.ema_initialized:
            self.ema.apply_shadow()

        pred = self.forward(image)
        loss = self.loss1(pred, label)

        if self.use_ema and self.ema_initialized:
            self.ema.restore()

        self.log("loss/valid_loss", loss, prog_bar=True)

        with torch.no_grad():
            pred_class = torch.argmax(pred, dim=1)
            accuracy = (pred_class == label).float().mean()
            self.log("accuracy", accuracy, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """训练轮次结束回调"""
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr)

    def on_validation_epoch_end(self):
        """验证轮次结束回调"""
        pass
