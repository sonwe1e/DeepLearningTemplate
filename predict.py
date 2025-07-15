import os
import torch
import torch.nn as nn
from PIL import Image
from configs.option import get_option, set_default_config_path
from tools.datasets.datasets import Dataset
from tools.model_registry import get_model
from tqdm import tqdm

# ==============================================================================
# 核心配置区域 (Core Configuration Area)
# ==============================================================================

# 1. 定义设备和TTA开关
# 1. Define device and TTA switch
DEVICE = torch.device("cuda:3")
USE_TTA = True  # 是否使用测试时间增强 (Whether to use Test-Time Augmentation)
SAVE_PATH = "/media/hdd/sonwe1e/DeepLearningTemplate/dataset/results"
os.makedirs(SAVE_PATH, exist_ok=True)

# 2. 定义模型集成配置
#   - 现在只需要指定权重文件路径，配置文件会自动从权重路径推导
#   - 默认配置文件位于权重所在实验目录的 'save_config.yaml'
# 2. Define Model Ensemble Configuration
#   - Now you only need to specify checkpoint paths, config paths will be inferred automatically
#   - Default config file is 'save_config.yaml' in the experiment directory containing the weights
MODELS_CONFIG = [
    "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/baseline-epoch300-5diceloss-convnext-rot2affine-randomwhitev4_2025-07-12_23-44-16/checkpoints/epoch_282-mIoU_0.575.ckpt",
    "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/baseline-epoch300-5diceloss-convnext-rot2affine-randomwhitev4_2025-07-12_23-44-16/checkpoints/epoch_257-mIoU_0.575.ckpt",
    "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/baseline-epoch300-5diceloss-convnext-rot2affine-randomwhitev4_2025-07-12_23-44-16/checkpoints/epoch_287-mIoU_0.575.ckpt",
]


# ==============================================================================
# 模型集成封装 (Model Ensemble Encapsulation)
# ==============================================================================


class ModelEnsemble:
    """
    一个管理和运行多个模型集成的类。
    它负责加载所有模型，并在推理时对它们的预测进行聚合。
    This class manages and runs an ensemble of multiple models.
    It is responsible for loading all models and aggregating their predictions at inference time.
    """

    def __init__(self, model_configs, device):
        """
        初始化函数，根据提供的配置加载所有模型。
        Initializes and loads all models based on the provided configurations.

        Args:
            model_configs (list): 模型权重路径列表，每个路径为一个字符串。
            device (torch.device): 模型将要加载到的设备 (e.g., torch.device("cuda:0"))。
        """
        self.models = []
        self.device = device
        print(f"Initializing Model Ensemble on device: {self.device}")

        for i, ckpt_path in enumerate(model_configs):
            print(f"--> Loading model {i + 1}/{len(model_configs)} from: {ckpt_path}")
            try:
                # 从权重路径推导配置文件路径
                # Infer config path from checkpoint path
                config_path = self._infer_config_path_from_checkpoint(ckpt_path)
                print(f"    Inferred config path: {config_path}")

                # 核心技巧：通过set_default_config_path临时设置全局配置路径，以便get_option能正确工作
                # Core trick: Temporarily set the global config path so get_option works correctly.
                set_default_config_path(config_path)
                opt, _ = get_option(verbose=False)

                # 加载模型架构
                # Load model architecture
                model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])

                # 加载权重
                # Load weights
                ckpt = torch.load(ckpt_path, map_location="cpu")

                # 检查并提取 state_dict (兼容多种保存格式)
                # Check for and extract the state_dict (compatible with various saving formats)
                state_dict = ckpt.get("ema_state_dict", ckpt.get("state_dict", ckpt))

                # 清理权重字典中可能存在的前缀 (如 'model.')
                # Clean potential prefixes (like 'model.') from the state dictionary keys
                cleaned_state_dict = {
                    k.replace("model.", "", 1): v for k, v in state_dict.items()
                }

                model.load_state_dict(cleaned_state_dict)
                model.to(self.device)
                model.eval()

                self.models.append(model)
                print(f"    Model {i + 1} loaded successfully.")
            except FileNotFoundError:
                print(
                    f"    [Warning] Could not find files for model {i + 1}: {ckpt_path} or its config. Skipping."
                )
            except Exception as e:
                print(
                    f"    [Error] Failed to load model {i + 1}. Reason: {e}. Skipping."
                )

        if not self.models:
            raise RuntimeError(
                "No models were successfully loaded into the ensemble. Please check your paths in MODELS_CONFIG."
            )
        print(f"\nEnsemble initialized with {len(self.models)} models.")

    def _infer_config_path_from_checkpoint(self, ckpt_path):
        """
        从权重文件路径推导出对应的配置文件路径。
        假设配置文件名为 'save_config.yaml' 且位于权重所在实验目录的根目录。

        Infers the configuration file path from the checkpoint file path.
        Assumes the config file is named 'save_config.yaml' and is located at the root of the experiment directory.

        Args:
            ckpt_path (str): 权重文件路径。

        Returns:
            str: 推导出的配置文件路径。
        """
        # 从路径中获取实验目录
        # 假设路径结构为 /path/to/experiments/experiment_name/checkpoints/model.ckpt
        experiment_dir = os.path.dirname(os.path.dirname(ckpt_path))
        config_path = os.path.join(experiment_dir, "save_config.yaml")

        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at inferred path: {config_path}"
            )

        return config_path

    def _perform_tta(self, model, sar, opt):
        """
        对单个模型执行测试时间增强（TTA）。
        Performs Test-Time Augmentation (TTA) for a single model.
        """
        # 存储所有增强结果的 logits
        # Store logits from all augmentations
        outputs_logits = []

        # 1. 原始预测 (Original)
        outputs_logits.append(model(sar, opt))

        # 2. 水平翻转 (Horizontal Flip)
        sar_hflip = torch.flip(sar, dims=[-1])
        opt_hflip = torch.flip(opt, dims=[-1])
        output_hflip = model(sar_hflip, opt_hflip)
        outputs_logits.append(torch.flip(output_hflip, dims=[-1]))

        # 3. 垂直翻转 (Vertical Flip)
        sar_vflip = torch.flip(sar, dims=[-2])
        opt_vflip = torch.flip(opt, dims=[-2])
        output_vflip = model(sar_vflip, opt_vflip)
        outputs_logits.append(torch.flip(output_vflip, dims=[-2]))

        # 组合操作可以提供更多视角，例如 180度旋转 等价于 水平+垂直翻转
        # Combined operations can provide more perspectives, e.g., 180-degree rotation is equivalent to h-flip + v-flip.
        # Here we add rotations for a more comprehensive D4 group augmentation.

        # 4. 90度旋转 (90-degree Rotation)
        sar_rot90 = torch.rot90(sar, k=1, dims=[-2, -1])
        opt_rot90 = torch.rot90(opt, k=1, dims=[-2, -1])
        output_rot90 = model(sar_rot90, opt_rot90)
        outputs_logits.append(torch.rot90(output_rot90, k=-1, dims=[-2, -1]))

        # 将所有TTA结果的logits堆叠起来，并在新维度上求平均
        # Stack the logits from all TTA results and average them across the new dimension.
        # 这是TTA的关键：在概率空间（或logits空间）进行平均，而不是在最终决策后投票。
        # This is the key to TTA: averaging in probability (or logit) space, not voting after the final decision.
        ensembled_tta_logits = torch.mean(torch.stack(outputs_logits), dim=0)
        return ensembled_tta_logits

    @torch.no_grad()
    def __call__(self, batch, use_tta=True):
        """
        执行集成推理。
        Executes ensemble inference.

        Args:
            batch (dict): 从dataloader获取的数据批次。
            use_tta (bool): 是否为每个模型应用TTA。

        Returns:
            torch.Tensor: 聚合后的最终预测 logits。
        """
        sar = batch["sar"].to(self.device)
        opt = batch["opt"].to(self.device)

        all_model_logits = []
        for model in self.models:
            if use_tta:
                model_logits = self._perform_tta(model, sar, opt)
            else:
                model_logits = model(sar, opt)
            all_model_logits.append(model_logits)

        # 聚合所有模型的logits：在模型维度上取平均
        # Aggregate logits from all models: average across the model dimension.
        # 这是模型集成的核心：平均模型输出的置信度，而不是最终的类别标签。
        # This is the core of model ensembling: averaging the confidence scores, not the final class labels.
        ensembled_logits = torch.mean(torch.stack(all_model_logits), dim=0)
        return ensembled_logits


# ==============================================================================
# 主执行逻辑 (Main Execution Logic)
# ==============================================================================


def main():
    # 1. 初始化模型集成
    # 1. Initialize the model ensemble
    ensemble = ModelEnsemble(MODELS_CONFIG, DEVICE)

    # 2. 创建验证数据集
    #    我们使用第一个模型的配置进行初始化
    #    假设所有模型都使用相同的数据格式和预处理。
    # 2. Create validation dataset
    #    We use the first model's config for initialization
    #    This assumes all models use the same data format and preprocessing.
    config_path = ensemble._infer_config_path_from_checkpoint(MODELS_CONFIG[0])
    set_default_config_path(config_path)
    opt, _ = get_option(verbose=False)
    valid_dataset = Dataset(
        phase="test",
        opt=opt,
        train_transform=None,
        valid_transform=None,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.valid_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    print("\nStarting ensemble inference...")
    # 3. 迭代数据并进行推理
    # 3. Iterate through data and perform inference
    for batch in tqdm(valid_dataloader, desc="Ensemble Prediction"):
        # 获取集成模型的聚合预测
        # Get the aggregated prediction from the ensemble
        ensembled_output_logits = ensemble(batch, use_tta=USE_TTA)

        # 输出结果处理
        # Post-process the output
        final_prediction = torch.argmax(ensembled_output_logits, dim=1)
        final_prediction = final_prediction.cpu().numpy().astype("uint8")

        # 保存预测结果
        # Save the prediction results
        for i in range(final_prediction.shape[0]):
            pred_image_path = os.path.join(SAVE_PATH, batch["name"][i])
            Image.fromarray(final_prediction[i]).save(pred_image_path, format="TIFF")

    print(f"\nInference complete. Results saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
