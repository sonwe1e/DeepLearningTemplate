import argparse
import yaml
from pathlib import Path
import os


def get_option(verbose=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "config.yaml")
    if Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
    else:
        yaml_config = {}

    parser = argparse.ArgumentParser()
    for key, value in yaml_config.items():
        if isinstance(value, bool):
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    # parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()
    # 合并命令行参数覆盖YAML配置
    final_config = dict(yaml_config)
    for k, v in vars(args).items():
        final_config[k] = v

    # 将合并后的配置保存回 config.yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(final_config, f, allow_unicode=True)

    if verbose:
        print("-" * 30)
        print("Current Configuration:")
        for k, v in final_config.items():
            print(f"{k}: {v}")
        print("-" * 30)

    return argparse.Namespace(**final_config)


if __name__ == "__main__":
    config = get_option()
    print(config)
