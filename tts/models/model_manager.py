import importlib
import logging
from typing import List
from .model_interface import TTSModelInterface


class ModelManager:
    def __init__(self):
        self._models = {}  # 模型名称到实例的映射

    def get_model(self, model_name: str) -> TTSModelInterface:
        """获取模型实例"""
        if model_name not in self._models:
            raise ValueError(f"模型{model_name}未加载")
        return self._models[model_name]

    def load_model(self, model_path: str):
        """动态加载并初始化模型"""
        try:
            module = importlib.import_module(f"tts.models.{model_path.lower()}.handler")
            model_class = getattr(module, f"TTSModel")
            if not issubclass(model_class, TTSModelInterface):
                raise TypeError(f"{model_path}TTSModel必须实现TTSModelInterface")
            model = model_class.create()  # 调用静态方法创建实例
            self._models[model.get_model_info().model_name] = model
        except (ImportError, AttributeError) as e:
            raise ValueError(f"加载模型{model_path}失败: {e}")

    def download_model(self, model_path: str) -> str:
        """下载模型"""
        try:
            module = importlib.import_module(f"tts.models.{model_path.lower()}.handler")
            model_class = getattr(module, f"TTSModel")
            if not issubclass(model_class, TTSModelInterface):
                raise TypeError(f"{model_path}TTSModel必须实现TTSModelInterface")
            path = model_class.download_model()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"下载模型{model_path}失败: {e}")
        return path

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())


model_manager = ModelManager()  # 全局实例
