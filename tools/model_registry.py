import os
import importlib
import inspect
from pathlib import Path
import torch.nn as nn
from typing import Dict, Type, Optional
from rich.console import Console
from rich.traceback import install

# richer traceback
# install(show_locals=True)


class ModelRegistry:
    """模型注册器，使用延迟导入策略避免循环导入"""

    def __init__(self):
        self._models: Dict[str, Type[nn.Module]] = {}
        self._discovered = False
        self._models_dir: Optional[Path] = None
        self.console = Console()

    def _get_models_dir(self) -> Path:
        """获取模型目录路径"""
        if self._models_dir is None:
            self._models_dir = Path(__file__).parent / "models"
        return self._models_dir

    def _discover_models(self):
        """延迟发现并注册所有模型"""
        if self._discovered:
            return

        models_dir = self._get_models_dir()

        if not models_dir.exists():
            self.console.print(
                f"[bold yellow]警告:[/bold yellow] 模型目录 [underline]{models_dir}[/underline] 不存在"
            )
            self._discovered = True
            return

        self.console.print(f"[bold cyan]开始发现模型目录:[/bold cyan] {models_dir}")

        for py_file in models_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                module_name = f"tools.models.{py_file.stem}"
                self.console.print(f"正在导入模块: [green]{module_name}[/green]")
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, nn.Module)
                        and obj is not nn.Module
                        and obj.__module__ == module_name
                    ):
                        model_key = name.lower()
                        self._models[model_key] = obj
                        self.console.print(
                            f"[green]发现模型:[/green] {name} -> [bold]{model_key}[/bold]"
                        )

            except Exception as e:
                self.console.print(f"[red]导入模型文件 {py_file} 时出错:[/red] {e}")
                raise  # rich.traceback 会自动格式化

        self._discovered = True
        self.console.print(
            f"[bold cyan]模型发现完成[/bold cyan]，共注册 [bold]{len(self._models)}[/bold] 个模型"
        )

    def get_model(self, model_name: str, **kwargs):
        """根据模型名称获取模型实例"""
        self._discover_models()
        key = model_name.lower()

        if key not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(f"未找到模型 '{model_name}'。可用模型: {available}")

        cls = self._models[key]
        try:
            self.console.print(
                f"[cyan]正在创建模型实例:[/cyan] {model_name} 参数: {kwargs}"
            )
            return cls(**kwargs)
        except Exception as e:
            self.console.print(f"[red]创建模型 '{model_name}' 时出错:[/red] {e}")
            raise RuntimeError(f"创建模型 '{model_name}' 时出错: {e}")

    def list_models(self):
        """列出所有可用的模型"""
        self._discover_models()
        return list(self._models.keys())

    def register_model(self, name: str, model_class: Type[nn.Module]):
        """手动注册模型"""
        self._models[name.lower()] = model_class
        self.console.print(f"[magenta]手动注册模型:[/magenta] {name} -> {name.lower()}")


# 全局单例
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_model(model_name: str, **kwargs):
    return get_model_registry().get_model(model_name, **kwargs)


def list_available_models():
    return get_model_registry().list_models()


def register_model(name: str, model_class: Type[nn.Module]):
    return get_model_registry().register_model(name, model_class)
