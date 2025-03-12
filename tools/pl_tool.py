import torch
from torchmetrics import ConfusionMatrix, F1Score
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
        self.ce_loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.train_preds = []  # 存储训练集预测结果
        self.train_labels = []  # 存储训练集标签
        self.valid_preds = []  # 存储验证集预测结果
        self.valid_labels = []  # 存储验证集标签
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=opt.num_classes
        )
        self.f1_score = F1Score(task="multiclass", num_classes=opt.num_classes)
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
            epochs=self.opt.epochs,
            steps_per_epoch=self.len_trainloader,
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
        ce_loss = self.ce_loss(prediction, label)  # 计算交叉熵损失
        loss = ce_loss
        self.train_preds.append(prediction)  # 存储预测值
        self.train_labels.append(label)  # 存储真实值
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
        ce_loss = self.ce_loss(prediction, label)  # 计算交叉熵损失
        loss = ce_loss
        self.valid_preds.append(prediction)  # 存储预测值
        self.valid_labels.append(label)  # 存储真实值
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
        train_preds = torch.cat(self.train_preds, 0)
        train_labels = torch.cat(self.train_labels, 0)
        preds = torch.argmax(train_preds, dim=1)  # 获取预测类别
        confusionmatrix = self.confusion_matrix(preds, train_labels)  # 计算混淆矩阵
        f1 = self.f1_score(preds, train_labels)  # 计算F1分数

        self.log(
            "metric/train_acc",
            confusionmatrix.diag().sum() / confusionmatrix.sum(),
        )
        self.log("metric/train_f1", f1)

        # 清空存储
        self.train_preds = []
        self.train_labels = []
        self.f1_score.reset()
        self.confusion_matrix.reset()

    def on_validation_epoch_end(self):
        """验证周期结束时执行"""
        valid_preds = torch.cat(self.valid_preds, 0)
        valid_labels = torch.cat(self.valid_labels, 0)
        preds = torch.argmax(valid_preds, dim=1)  # 获取预测类别
        confusionmatrix = self.confusion_matrix(preds, valid_labels)  # 计算混淆矩阵
        f1 = self.f1_score(preds, valid_labels)  # 计算F1分数

        self.log(
            "metric/valid_acc",
            confusionmatrix.diag().sum() / confusionmatrix.sum(),
        )
        self.log("metric/valid_f1", f1)  # 记录整体F1分数

        # 清空存储
        self.valid_preds = []
        self.valid_labels = []
        self.f1_score.reset()
        self.confusion_matrix.reset()

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
