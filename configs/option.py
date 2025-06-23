import argparse
import yaml
from pathlib import Path
import os
from datetime import datetime

# 全局变量用于存储默认配置路径
_DEFAULT_CONFIG_PATH = None


def set_default_config_path(path: str):
    """设置全局默认配置路径"""
    global _DEFAULT_CONFIG_PATH
    _DEFAULT_CONFIG_PATH = path


def get_option(
    default_config_path: str = None,
    output_root: str = "experiments",
    verbose: bool = True,
):
    """
    加载、合并和保存实验配置。

    处理流程:
    1. 从 default_config_path (如 'config.yaml') 加载基线配置。
    2. 基于基线配置动态构建命令行参数解析器 (argparse)。
    3. 解析命令行传入的参数。
    4. 将命令行参数覆盖到基线配置上，形成最终的有效配置。
    5. 基于项目名(project)和实验名(exp_name)创建一个带时间戳的唯一实验目录。
    6. 将最终有效配置保存到该实验目录中 (如 'experiments/Test/baselinev1/2023-10-27_10-30-00/effective_config.yaml')。
    7. 返回包含最终配置的 Namespace 对象和实验目录路径。

    Args:
        default_config_path (str): 默认配置文件的路径。
        output_root (str): 所有实验输出的根目录。
        verbose (bool): 是否打印最终配置。

    Returns:
        tuple[argparse.Namespace, Path]: (最终配置, 本次运行的实验目录路径)
    """
    # 优先级：函数参数 > 全局变量 > 环境变量 > 默认值
    if default_config_path is None:
        default_config_path = (
            _DEFAULT_CONFIG_PATH or os.environ.get("CONFIG_PATH") or "config.yaml"
        )

    # --- 1. 加载基线配置 ---
    config_path_obj = Path(default_config_path)
    if config_path_obj.exists():
        with open(config_path_obj, "r", encoding="utf-8") as f:
            # 使用 ruamel.yaml 可以保留注释，但这里我们只需要读取，所以 pyyaml 也可以
            # 为了简单起见，我们继续使用 pyyaml，因为我们不写回原文件
            yaml_config = yaml.safe_load(f) or {}
    else:
        # 如果没有配置文件，就不能动态生成 parser，这是一个严重的问题
        raise FileNotFoundError(f"默认配置文件未找到: {default_config_path}")

    # --- 2. 基于基线配置构建解析器 ---
    parser = argparse.ArgumentParser(description="实验配置")
    # 添加一个指向原始配置文件的参数，以便记录
    parser.add_argument(
        "--config", type=str, default=default_config_path, help="指向原始配置模板的路径"
    )

    for key, value in yaml_config.items():
        # 这个布尔逻辑是正确的：
        # 如果默认是 True，我们提供一个 store_false 的 flag (例如 --no-ema)
        # 如果默认是 False，我们提供一个 store_true 的 flag (例如 --use-amp)
        # 更好的做法是使用 BooleanOptionalAction (Python 3.9+)，但当前实现兼容性更广
        if isinstance(value, bool):
            if value:
                parser.add_argument(
                    f"--{key}", action="store_false", dest=key
                )  # 使用 dest 确保覆盖正确的键
            else:
                parser.add_argument(f"--{key}", action="store_true", dest=key)
        else:
            # 对于列表类型，nargs='+' 或 '*' 更好，但为了保持与原逻辑一致，暂不修改
            parser.add_argument(
                f"--{key}", type=type(value), default=None
            )  # Default=None 以便区分未指定和指定

    # --- 3. 解析命令行参数 ---
    args = parser.parse_args()

    # --- 4. 合并配置 (命令行 > YAML) ---
    final_config = dict(yaml_config)
    for k, v in vars(args).items():
        # 仅当命令行中明确指定了该参数时才进行覆盖
        # argparse 对于未指定的参数会赋予 default 值 (我们设为 None)
        if v is not None:
            final_config[k] = v

    # --- 5. 创建唯一的实验目录 ---
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = final_config.get("exp_name", "default-exp")

        exp_path = Path(output_root) / f"{exp_name}_{timestamp}"
        exp_path.mkdir(parents=True, exist_ok=True)

        # --- 6. 将最终有效配置保存到新目录 ---
        # 这是关键的修改：写入到新的、本次运行专属的配置文件中
        effective_config_path = exp_path / "save_config.yaml"
        with open(effective_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(final_config, f, allow_unicode=True, sort_keys=False)

        # --- 7. 打印并返回 ---
        print("=" * 40)
        print(f"原始配置文件: {config_path_obj.resolve()}")
        print(f"实验输出目录: {exp_path.resolve()}")
        print("-" * 40)
        print("最终生效配置 (Final Effective Configuration):")
        for k, v in final_config.items():
            print(f"{k:<20}: {v}")
        print("=" * 40)

        # 添加实验路径到配置中，方便后续代码（如模型保存）直接使用
        final_config["exp_path"] = str(exp_path)

    return argparse.Namespace(**final_config), exp_path if verbose else None


if __name__ == "__main__":
    # 模拟命令行调用: python your_script.py --learning_rate 0.001 --epochs 50
    # 在实际运行时，这些参数会由 sys.argv 提供

    # 首先，确保我们有一个 config.yaml 文件用于测试
    config_content = """
# ===================================================================
# 实验管理与日志 (Experiment Management & Logging)
# ===================================================================
project: Test
exp_name: baselinev1
seed: 42

# ===================================================================
# 模型定义 (Model Definition)
# ===================================================================
model_name: resnet18d.ra2_in1k
pretrained: true
num_classes: 3

# ===================================================================
# 优化器与学习率调度 (Optimizer & LR Scheduler)
# ===================================================================
learning_rate: 0.0004
epochs: 100
    """
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)

    print(">>> 运行 get_option...")
    # 在脚本中，直接调用即可
    config = get_option()

    print("\n>>> 返回的 config 对象:")
    print(config)
    print(f"\n可以通过 config.exp_path 访问实验目录: {config.exp_path}")

    # 清理测试文件和目录
    # os.remove("config.yaml")
    # import shutil
    # if os.path.exists("experiments"):
    #     shutil.rmtree("experiments")
