import torch
import lightning.pytorch as pl
from .model_registry import get_model
import torch.nn.functional as F
from tools.losses import DiceLoss, IdentityLoss, LovaszSoftmax, FocalLoss

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
        self.layer_decay = opt.layer_decay
        self.len_trainloader = len_trainloader
        self.opt = opt

        self.model = model

        self.loss1 = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss2 = DiceLoss()
        self.loss3 = LovaszSoftmax()

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
        if self.layer_decay <= 0.0 or self.layer_decay >= 1.0:
            # 如果没有使用层级学习率衰减，使用原来的优化器配置
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                weight_decay=self.opt.weight_decay,
                lr=self.learning_rate,
            )
        else:
            # 使用层级学习率衰减
            param_groups = self._get_layer_wise_lr_param_groups()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.opt.weight_decay,
            )

        # 获取实际的步数，考虑多卡并行训练
        if hasattr(self.trainer, "world_size") and self.trainer.world_size > 1:
            total_steps = (
                self.len_trainloader // self.trainer.world_size
            ) * self.opt.epochs
        else:
            total_steps = self.len_trainloader * self.opt.epochs

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=self.opt.pct_start,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def _get_layer_wise_lr_param_groups(self):
        """
        按照层次划分参数组，并应用层级学习率衰减
        """
        # 基于模型类型判断层次，这里以典型的backbone+decoder为例
        param_groups = []

        # 如果是类似DualStreamTimmUnet的模型，可以分层次应用衰减
        if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
            # 1. 分离backbone参数（encoder）
            encoder_layers = []
            if hasattr(self.model.encoder, "sar_encoder") and hasattr(
                self.model.encoder.sar_encoder, "stages"
            ):
                # 将encoder分为不同阶段
                for i, stage in enumerate(self.model.encoder.sar_encoder.stages):
                    decay = self.layer_decay ** (
                        len(self.model.encoder.sar_encoder.stages) - i
                    )
                    encoder_layers.append(
                        {
                            "params": stage.parameters(),
                            "lr": self.learning_rate * decay,
                            "name": f"sar_encoder_stage_{i}",
                        }
                    )

            if hasattr(self.model.encoder, "opt_encoder") and hasattr(
                self.model.encoder.opt_encoder, "stages"
            ):
                for i, stage in enumerate(self.model.encoder.opt_encoder.stages):
                    decay = self.layer_decay ** (
                        len(self.model.encoder.opt_encoder.stages) - i
                    )
                    encoder_layers.append(
                        {
                            "params": stage.parameters(),
                            "lr": self.learning_rate * decay,
                            "name": f"opt_encoder_stage_{i}",
                        }
                    )

            # 2. 融合模块参数
            if hasattr(self.model.encoder, "fusion_modules"):
                param_groups.append(
                    {
                        "params": self.model.encoder.fusion_modules.parameters(),
                        "lr": self.learning_rate,
                        "name": "fusion_modules",
                    }
                )

            # 3. 解码器参数（通常使用更高的学习率）
            param_groups.append(
                {
                    "params": self.model.decoder.parameters(),
                    "lr": self.learning_rate,
                    "name": "decoder",
                }
            )

            # 添加encoder层参数组
            param_groups.extend(encoder_layers)
        else:
            # 默认情况下，对整个模型使用相同的学习率
            param_groups.append(
                {
                    "params": self.model.parameters(),
                    "lr": self.learning_rate,
                    "name": "whole_model",
                }
            )

        return param_groups

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
        lovasz_loss = self.loss3(prediction, label)
        loss = ce_loss + 5 * dice_loss + lovasz_loss

        ious = self._calculate_miou(prediction, label, self.num_classes)
        self.train_iou_sum += ious
        self.train_iou_count += 1

        self.log("loss/train_ce_loss", ce_loss)
        self.log("loss/train_lovasz_loss", lovasz_loss)
        self.log("loss/train_dice_loss", dice_loss)
        self.log("loss/train_loss", loss, prog_bar=True)
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        if self.use_ema:
            self.ema.update()
        return loss

    def validation_step(self, batch, batch_idx):
        sar, opt, label = (batch["sar"], batch["opt"], batch["label"])
        prediction = self(sar, opt)
        ce_loss = self.loss1(prediction, label)
        dice_loss = self.loss2(prediction, label)
        lovasz_loss = self.loss3(prediction, label)
        loss = ce_loss + dice_loss + lovasz_loss

        ious = self._calculate_miou(prediction, label, self.num_classes)
        self.val_iou_sum += ious
        self.val_iou_count += 1

        self.log("loss/valid_ce_loss", ce_loss)
        self.log("loss/valid_dice_loss", dice_loss)
        self.log("loss/valid_lovasz_loss", lovasz_loss)
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
