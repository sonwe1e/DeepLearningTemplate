import torch
import lightning.pytorch as pl
from .model_registry import get_model
import torch.nn.functional as F
from tools.losses import DiceLoss

torch.set_float32_matmul_precision("high")


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.data.dtype)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt

        if model is None:
            model_kwargs = {
                "input_channels": getattr(opt, "in_chans", 3),
                "num_classes": getattr(opt, "num_classes", 10),
            }
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            self.model = get_model(opt.model_name, **model_kwargs)
            print(f"动态加载模型: {opt.model_name}")
        else:
            self.model = model

        self.loss1 = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss2 = DiceLoss()

        self.use_ema = getattr(opt, "use_ema", True)
        self.ema_decay = getattr(opt, "ema_decay", 0.999)
        if self.use_ema:
            self.ema = None
            self.ema_initialized = False

        self.num_classes = getattr(opt, "num_classes", 6)
        self.train_iou_sum = torch.zeros(self.num_classes)
        self.train_iou_count = 0
        self.val_iou_sum = torch.zeros(self.num_classes)
        self.val_iou_count = 0

    def _init_ema(self):
        if self.use_ema and not self.ema_initialized:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema_initialized = True

    def forward(self, sar, opt):
        pred = self.model(sar, opt)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.len_trainloader * self.opt.epochs,
            pct_start=self.opt.pct_start,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def _calculate_miou(self, pred, target, num_classes):
        pred = torch.argmax(pred, dim=1)
        ious = []

        for cls in range(num_classes):
            pred_cls = pred == cls
            target_cls = target == cls

            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            if union == 0:
                ious.append(1.0)
            else:
                ious.append((intersection / union).item())

        return torch.tensor(ious)

    def training_step(self, batch, batch_idx):
        if self.use_ema and not self.ema_initialized:
            self._init_ema()
        sar, opt, label = (batch["sar"], batch["opt"], batch["label"])
        prediction = self(sar, opt)
        ce_loss = self.loss1(prediction, label)
        dice_loss = self.loss2(prediction, label)
        loss = ce_loss + dice_loss

        ious = self._calculate_miou(prediction, label, self.num_classes)
        self.train_iou_sum += ious
        self.train_iou_count += 1

        self.log("loss/train_ce_loss", ce_loss)
        self.log("loss/train_loss", loss)
        self.log("loss/train_dice_loss", dice_loss)
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        if self.use_ema:
            self.ema.update()
        return loss

    def validation_step(self, batch, batch_idx):
        sar, opt, label = (batch["sar"], batch["opt"], batch["label"])
        prediction = self(sar, opt)
        ce_loss = self.loss1(prediction, label)
        dice_loss = self.loss2(prediction, label)
        loss = ce_loss + dice_loss

        ious = self._calculate_miou(prediction, label, self.num_classes)
        self.val_iou_sum += ious
        self.val_iou_count += 1

        self.log("loss/valid_ce_loss", ce_loss, prog_bar=True)
        self.log("loss/valid_dice_loss", dice_loss, prog_bar=True)
        self.log("loss/valid_loss", loss)

    def on_validation_start(self):
        if self.use_ema and self.ema_initialized:
            self.ema.apply_shadow()

    def on_validation_end(self):
        if self.use_ema and self.ema_initialized:
            self.ema.restore()

    def on_train_epoch_end(self):
        if self.train_iou_count > 0:
            mean_ious = self.train_iou_sum / self.train_iou_count
            miou = mean_ious.mean()
            self.log("metrics/train_miou", miou)
            for i, iou in enumerate(mean_ious):
                self.log(f"metrics/train_iou_class_{i}", iou)

            self.train_iou_sum = torch.zeros(self.num_classes)
            self.train_iou_count = 0

    def on_validation_epoch_end(self):
        if self.val_iou_count > 0:
            mean_ious = self.val_iou_sum / self.val_iou_count
            miou = mean_ious.mean()
            self.log("metrics/val_miou", miou, prog_bar=True)
            for i, iou in enumerate(mean_ious):
                self.log(f"metrics/val_iou_class_{i}", iou)

            self.val_iou_sum = torch.zeros(self.num_classes)
            self.val_iou_count = 0

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema and self.ema_initialized:
            checkpoint["ema_state_dict"] = {
                k: v.clone().float().cpu() for k, v in self.ema.shadow.items()
            }

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and "ema_state_dict" in checkpoint:
            self._init_ema()
            for k, v in checkpoint["ema_state_dict"].items():
                if k in self.ema.shadow:
                    self.ema.shadow[k] = v.to(self.device)
