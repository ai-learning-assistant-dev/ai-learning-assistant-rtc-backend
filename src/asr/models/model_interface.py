from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

import numpy as np
from pydantic import BaseModel


class ModelDetail(BaseModel):
    model_name: str
    description: str
    device: str
    sample_rate: int

class Timestamp(BaseModel):
    text: str
    start: int
    end: int
    words: List

class TranscribeData(BaseModel):
    text: str
    segments: List[Timestamp]
    language: str


class ASRModelInterface(ABC):
    @abstractmethod
    def raw_transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """
        识别语音并返回已识别的字符串
        """
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> TranscribeData:
        """
        识别语音并返回已识别的字符串
        """
        pass

    @staticmethod
    @abstractmethod
    def create(vad_model: str | None = None) -> "ASRModelInterface":
        """初始化并传入VAD模型的本地路径

        Returns:
            ASRModelInterface: 初始化完成的ASR模型实例
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


class AsyncASRModelInterface(ABC):
    @abstractmethod
    async def stream_transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        异步流式合成语音并返回音频数据块
        Returns:
            audio_chunk: numpy数组格式的音频波形数据块
        """
        # 如果不这样写，函数返回值类型会变成Coroutine
        if False:
            yield ""  # type: ignore
        # 但是实际实现中必须重写此方法
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create() -> "AsyncASRModelInterface":
        """Initialize the model with configuration.

        Returns:
            AsyncASRModelInterface: 初始化完成的RTC ASR模型实例
        """
        pass
