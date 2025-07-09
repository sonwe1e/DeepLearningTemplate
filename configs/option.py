import argparse
import yaml
from pathlib import Path
import os
from datetime import datetime

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
    """加载、合并和保存实验配置"""
    if default_config_path is None:
        default_config_path = (
            _DEFAULT_CONFIG_PATH or os.environ.get("CONFIG_PATH") or "config.yaml"
        )

    config_path_obj = Path(default_config_path)
    if config_path_obj.exists():
        with open(config_path_obj, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"默认配置文件未找到: {default_config_path}")

    parser = argparse.ArgumentParser(description="实验配置")
    parser.add_argument(
        "--config", type=str, default=default_config_path, help="配置文件路径"
    )

    for key, value in yaml_config.items():
        if isinstance(value, bool):
            if value:
                parser.add_argument(f"--{key}", action="store_false", dest=key)
            else:
                parser.add_argument(f"--{key}", action="store_true", dest=key)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=None)

    args = parser.parse_args()

    final_config = dict(yaml_config)
    for k, v in vars(args).items():
        if v is not None:
            final_config[k] = v

    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = final_config.get("exp_name", "default-exp")
        exp_path = Path(output_root) / f"{exp_name}_{timestamp}"
        exp_path.mkdir(parents=True, exist_ok=True)

        config_save_path = exp_path / "save_config.yaml"
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(final_config, f, default_flow_style=False, allow_unicode=True)

        if verbose:
            print(f"实验目录: {exp_path}")
            print(f"配置已保存到: {config_save_path}")

        final_config = argparse.Namespace(**final_config)
        return final_config, exp_path
    else:
        final_config = argparse.Namespace(**final_config)
        return final_config, None
