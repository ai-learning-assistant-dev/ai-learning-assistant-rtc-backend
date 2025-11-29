import importlib
from typing import List
from .model_interface import ASRModelInterface


class ModelManager:
    def __init__(self):
        self._models = {}  # 模型名称到实例的映射

    def get_model(self, model_name: str) -> ASRModelInterface | None:
        """获取模型实例"""
        if model_name not in self._models:
            return None
        return self._models[model_name]

    def load_model(self, model_name: str):
        """动态加载并初始化模型"""
        try:
            module = importlib.import_module(f"asr.models.{model_name.lower()}.handler")
            model_class = getattr(module, "ASRModel")
            if not issubclass(model_class, ASRModelInterface):
                raise TypeError(f"{model_name}ASRModel必须实现ASRModelInterface")
            model = model_class.create()  # 调用静态方法创建实例
            self._models[model.get_model_info().model_name] = model
        except (ImportError, AttributeError) as e:
            raise ValueError(f"加载模型{model_name}失败: {e}")
    
    def download_model(self, model_name: str) -> str:
        """下载模型"""
        try:
            module = importlib.import_module(f"asr.models.{model_name.lower()}.handler")
            model_class = getattr(module, "ASRModel")
            if not issubclass(model_class, ASRModelInterface):
                raise TypeError(f"{model_name}ASRModel必须实现ASRModelInterface")
            path = model_class.download_model()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"下载模型{model_name}失败: {e}")
        return path

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())


asr_model_manager = ModelManager()  # 全局实例
