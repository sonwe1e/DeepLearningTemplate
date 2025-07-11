# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d
import timm
from typing import List, Dict, Optional, Tuple

# ======================================================================================
# 1. Core Building Blocks & Modules (保持不变)
# ======================================================================================


class SelectiveFeatureFusion(nn.Module):
    """
    通过动态通道注意力机制，自适应地融合多个并行的特征分支。
    """

    def __init__(self, in_channels: int, num_branches: int, kernel_size: int = 3):
        super().__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_generator = nn.Conv1d(
            1, num_branches, kernel_size, padding=kernel_size // 2
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_branches: List[torch.Tensor]) -> torch.Tensor:
        if len(feature_branches) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} feature branches, but got {len(feature_branches)}"
            )

        B, C, H, W = feature_branches[0].shape
        stacked_features = torch.stack(feature_branches, dim=1)
        fused_features = torch.sum(stacked_features, dim=1)
        pooled_features = self.global_avg_pool(fused_features)
        attention_input = pooled_features.squeeze(-1).permute(0, 2, 1)
        attention_scores = self.attention_generator(attention_input)
        attention_weights = self.softmax(attention_scores).unsqueeze(-1).unsqueeze(-1)
        output = torch.sum(stacked_features * attention_weights, dim=1)
        return output


class MultiScaleConvHead(nn.Module):
    """
    一个多尺度卷积头，用于并行提取不同感受野的特征，并使用注意力机制进行融合。
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels

        kernel_sizes = [1, 3, 5, 7]
        self.num_branches = len(kernel_sizes)

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )

        self.attention_fusion = SelectiveFeatureFusion(
            in_channels=in_channels,
            num_branches=self.num_branches,
            kernel_size=7,
        )

        self.projection_conv = nn.Conv2d(
            in_channels * (self.num_branches + 1), num_classes, kernel_size=7, padding=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_features = [conv(x) for conv in self.conv_branches]
        fused_feature = self.attention_fusion(multi_scale_features)
        all_features = torch.cat(multi_scale_features + [fused_feature], dim=1)
        output = self.projection_conv(all_features)
        return output


class DualPathConvBlock(nn.Module):
    """
    一个具有两个并行卷积路径和残差连接的自定义卷积块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.has_residual = in_channels == out_channels
        hidden_channels = int(in_channels * expand_ratio)

        self.path1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
            ),
            LayerNorm2d(in_channels),
            nn.Dropout(dropout_rate / 2),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        self.path2 = nn.Sequential(
            nn.Dropout(dropout_rate / 2),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                groups=hidden_channels,
            ),
            LayerNorm2d(hidden_channels),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.path1(x) + self.path2(x)
        if self.has_residual:
            output += x
        return output


# ======================================================================================
# 2. Feature Fusion Module
# ======================================================================================


class CrossModalFeatureFusion(nn.Module):
    """
    跨模态特征融合模块，用于融合SAR和OPT的特征。
    支持多种融合策略：add, concat, attention
    """

    def __init__(self, channels: int, fusion_type: str = "attention"):
        super().__init__()
        self.fusion_type = fusion_type
        self.channels = channels

        if fusion_type == "attention":
            # 使用注意力机制进行融合
            self.attention_fusion = SelectiveFeatureFusion(
                in_channels=channels, num_branches=2, kernel_size=3
            )
        elif fusion_type == "concat":
            # 拼接后降维
            self.fusion_conv = nn.Conv2d(
                channels * 2, channels, kernel_size=1, bias=False
            )
            self.norm = LayerNorm2d(channels)
        # "add" 模式不需要额外参数

    def forward(self, sar_feat: torch.Tensor, opt_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "add":
            return sar_feat + opt_feat
        elif self.fusion_type == "concat":
            fused = torch.cat([sar_feat, opt_feat], dim=1)
            fused = self.fusion_conv(fused)
            return self.norm(fused)
        elif self.fusion_type == "attention":
            return self.attention_fusion([sar_feat, opt_feat])
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


# ======================================================================================
# 3. Dual-Stream Encoder
# ======================================================================================


class DualStreamEncoder(nn.Module):
    """
    双流编码器，为SAR和OPT数据提供独立的编码路径。
    在每个层级进行特征融合，然后传递给下一层。
    """

    def __init__(
        self,
        model_config: str,
        encoder_channels: List[int],
        fusion_type: str = "attention",
        pretrained: bool = True,
    ):
        super().__init__()

        # 创建SAR编码器 (3通道输入)
        self.sar_encoder = timm.create_model(
            model_config, pretrained=pretrained, features_only=True, in_chans=3
        )

        # 创建OPT编码器 (3通道输入，但会复制1通道到3通道)
        self.opt_encoder = timm.create_model(
            model_config, pretrained=pretrained, features_only=True, in_chans=3
        )

        # 创建每个层级的特征融合模块
        self.fusion_modules = nn.ModuleList(
            [
                CrossModalFeatureFusion(channels, fusion_type)
                for channels in encoder_channels
            ]
        )

        self.encoder_channels = encoder_channels

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            sar: SAR数据 [B, 3, H, W]
            opt: OPT数据 [B, 1, H, W]

        Returns:
            List of fused features from each encoder stage
        """
        # 将OPT的单通道复制为3通道
        opt_3ch = opt.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]

        # 分别通过两个编码器
        sar_features = self.sar_encoder(sar)
        opt_features = self.opt_encoder(opt_3ch)

        # 在每个层级进行特征融合
        fused_features = []
        for i, (sar_feat, opt_feat, fusion_module) in enumerate(
            zip(sar_features, opt_features, self.fusion_modules)
        ):
            fused_feat = fusion_module(sar_feat, opt_feat)
            fused_features.append(fused_feat)

        return fused_features


# ======================================================================================
# 4. Modified UNet Decoder
# ======================================================================================


class UNetDecoder(nn.Module):
    """
    标准的U-Net解码器，接收融合后的特征进行解码。
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
        fusion_mode: str = "add",
    ):
        super().__init__()
        if fusion_mode not in ["add", "concat"]:
            raise ValueError(
                f"Fusion mode must be 'add' or 'concat', but got '{fusion_mode}'"
            )
        self.fusion_mode = fusion_mode

        encoder_channels = encoder_channels[::-1]

        self.upsample_layers = nn.ModuleList()
        self.fusion_convs = nn.ModuleList() if fusion_mode == "concat" else None

        for i in range(len(decoder_channels)):
            up_conv = nn.ConvTranspose2d(
                in_channels=encoder_channels[i],
                out_channels=decoder_channels[i],
                kernel_size=2,
                stride=2,
            )
            self.upsample_layers.append(up_conv)

            if fusion_mode == "concat":
                fusion_in_channels = decoder_channels[i] + encoder_channels[i + 1]
                fusion_out_channels = decoder_channels[i]
                self.fusion_convs.append(
                    nn.Conv2d(
                        fusion_in_channels,
                        fusion_out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )

        self.segmentation_head = nn.Sequential(
            nn.Dropout(0.2), MultiScaleConvHead(decoder_channels[-1], num_classes)
        )

    def forward(
        self, encoder_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = encoder_features[-1]
        skip_connections = encoder_features[::-1][1:]

        for i in range(len(self.upsample_layers)):
            x = self.upsample_layers[i](x)
            skip = skip_connections[i]

            if self.fusion_mode == "concat":
                x = torch.cat([x, skip], dim=1)
                x = self.fusion_convs[i](x)
            elif self.fusion_mode == "add":
                x = x + skip

        output = self.segmentation_head(x)
        return output, x


# ======================================================================================
# 5. Main Architecture: Dual-Stream U-Net
# ======================================================================================


class DualStreamTimmUnet(nn.Module):
    """
    双流U-Net架构，分别处理SAR和OPT数据，在编码器的每个层级进行特征融合。
    """

    MODEL_CHANNELS_MAP: Dict[str, List[int]] = {
        "rdnet_large.nv_in1k_ft_in1k_384": [528, 840, 1528, 2000],
        "efficientvit_l3.r384_in1k": [128, 256, 512, 1024],
        "rdnet_base.nv_in1k": [408, 584, 1000, 1760],
        "rdnet_small.nv_in1k": [264, 512, 760, 1264],
        "rdnet_tiny.nv_in1k": [256, 440, 744, 1040],
    }

    def __init__(
        self,
        model_config: str,
        num_classes: int,
        fusion_mode: str = "add",  # 解码器的融合模式
        cross_modal_fusion: str = "attention",  # 跨模态融合模式
        pretrained: bool = True,
    ):
        super().__init__()

        if model_config not in self.MODEL_CHANNELS_MAP:
            raise KeyError(
                f"Model '{model_config}' not found in MODEL_CHANNELS_MAP. "
                f"Please add its channel configuration."
            )

        encoder_channels = self.MODEL_CHANNELS_MAP[model_config]

        # 1. 创建双流编码器
        self.encoder = DualStreamEncoder(
            model_config=model_config,
            encoder_channels=encoder_channels,
            fusion_type=cross_modal_fusion,
            pretrained=pretrained,
        )

        # 2. 创建解码器
        decoder_channels = encoder_channels[::-1][1:]
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            fusion_mode=fusion_mode,
        )

        self.model_config = model_config

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar: SAR数据 [B, 3, H, W]
            opt: OPT数据 [B, 1, H, W]

        Returns:
            logits: 分割预测 [B, num_classes, H, W]
        """
        input_size = sar.shape[2:]

        # 1. 双流编码器前向传播，获取融合后的多尺度特征图
        fused_features = self.encoder(sar, opt)

        # 2. 解码器前向传播，得到分割图的logits
        logits, last_feature = self.decoder(fused_features)

        # 3. 将输出上采样到原始输入尺寸
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        return logits

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        加载预训练权重的辅助函数。
        """
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 移除 'model.' 前缀（如果存在）
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        self.load_state_dict(
            clean_state_dict, strict=False
        )  # strict=False for partial loading
        print("Weights loaded successfully.")

    def load_encoder_weights_separately(self, sar_checkpoint: str, opt_checkpoint: str):
        """
        分别加载SAR和OPT编码器的预训练权重。
        """
        print(f"Loading SAR encoder weights from: {sar_checkpoint}")
        sar_state = torch.load(sar_checkpoint, map_location="cpu")
        sar_state_dict = sar_state.get("state_dict", sar_state)

        print(f"Loading OPT encoder weights from: {opt_checkpoint}")
        opt_state = torch.load(opt_checkpoint, map_location="cpu")
        opt_state_dict = opt_state.get("state_dict", opt_state)

        # 加载到对应的编码器中
        self.encoder.sar_encoder.load_state_dict(sar_state_dict, strict=False)
        self.encoder.opt_encoder.load_state_dict(opt_state_dict, strict=False)

        print("Encoder weights loaded successfully.")


# ======================================================================================
# 6. Execution Example
# ======================================================================================

if __name__ == "__main__":
    # --- 设定参数 ---
    MODEL_CONFIG = "rdnet_small.nv_in1k"
    NUM_CLASSES = 1  # 二分类分割
    FUSION_MODE = "add"  # 解码器融合模式
    CROSS_MODAL_FUSION = "attention"  # 跨模态融合模式
    BATCH_SIZE = 2
    IMAGE_SIZE = (512, 512)  # 使用较小尺寸进行测试

    # --- 创建模型实例 ---
    model = DualStreamTimmUnet(
        model_config=MODEL_CONFIG,
        num_classes=NUM_CLASSES,
        fusion_mode=FUSION_MODE,
        cross_modal_fusion=CROSS_MODAL_FUSION,
        pretrained=True,  # 在测试时不下载预训练权重
    )

    # --- 创建模拟输入张量 ---
    sar_input = torch.randn(BATCH_SIZE, 1, *IMAGE_SIZE)  # SAR: 3通道
    opt_input = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE)  # OPT: 1通道

    # --- 模型前向传播 ---
    print(f"SAR input shape: {sar_input.shape}")
    print(f"OPT input shape: {opt_input.shape}")

    try:
        logits = model(sar_input, opt_input)
        print(f"Output logits shape: {logits.shape}")

        # --- 验证输出形状 ---
        expected_shape = (BATCH_SIZE, NUM_CLASSES, *IMAGE_SIZE)
        assert logits.shape == expected_shape, (
            f"Shape mismatch! Expected {expected_shape}, but got {logits.shape}"
        )

        print(
            "\nDual-stream model forward pass successful and output shape is correct."
        )

        # --- 打印模型参数统计 ---
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"\nAn error occurred during model forward pass: {e}")
        import traceback

        traceback.print_exc()
