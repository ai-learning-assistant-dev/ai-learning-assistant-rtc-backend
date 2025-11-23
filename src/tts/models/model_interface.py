# models/model_interface.py
import typing
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Callable, List, Union
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
    sample_rate: int


class TTSModelInterface(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_type: str,
        speed: Union[float, Callable[[int], float]] = 1,
    ) -> np.ndarray:
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


class RTCTTSModelInterface(TTSModelInterface):
    @abstractmethod
    async def stream_synthesize(
        self,
        text: str,
        voice_type: str,
        speed: Union[float, Callable[[int], float]] = 1,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        异步流式合成语音并返回音频数据块
        Returns:
            audio_chunk: numpy数组格式的音频波形数据块
        """
        # 如果不这样写，函数返回值类型会变成Coroutine
        if False:
            yield np.array([], dtype=np.float32)  # type: ignore
        # 但是实际实现中必须重写此方法
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create() -> "RTCTTSModelInterface":
        """Initialize the model with configuration.

        Returns:
            RTCTTSModelInterface: 初始化完成的RTC TTS模型实例
        """
        pass
