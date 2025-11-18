# models/model_interface.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pydantic import BaseModel


class VoiceDetail(BaseModel):
    name: str
    description: str


class ModelDetail(BaseModel):
    model_name: str
    description: str
    device: str
    voices: List[VoiceDetail]
    max_input_length: int


class TTSModelInterface(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice_type: str, speed: float) -> np.ndarray:
        """
        合成语音并返回音频数据
        Returns:
            audio_data: numpy数组格式的音频波形数据
        """
        pass

    @staticmethod
    @abstractmethod
    def create() -> "TTSModelInterface":
        """Initialize the model with configuration.

        Returns:
            TTSModelInterface: 初始化完成的TTS模型实例
        """
        pass

    @staticmethod
    @abstractmethod
    def download_model() -> str:
        """Download model resources."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelDetail:
        """Get model information."""
        pass

    @abstractmethod
    def max_input_length(self) -> int:
        """Get the maximum input length for the model."""
        pass
